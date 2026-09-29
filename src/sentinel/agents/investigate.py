"""The Investigation Agent (PRD F-05, Section 5.5.6).

F-05's acceptance criterion is absolute: *"every factual claim traces to a
retrieved KB chunk or raw log line"*. Part 1 made that structural —
:class:`~sentinel.core.schemas.InvestigationReport` rejects a claim with no
citations, and rejects a citation whose ref is not in the report's own evidence
list. This module's job is to make the *contents* of that list real, which is a
different and harder problem: a report can satisfy the schema perfectly while
citing five chunks that say nothing about the alert.

Lookup first, search second
---------------------------
The Triage Agent has already named a technique, and that name is an *identifier*,
not a query. So the report is built from a direct lookup of that technique's
chunks, plus the knowledge base's own relation graph (the CVEs that realise it,
the playbooks that respond to it), and only then from a text search for context
the identifier does not reach.

This ordering matters more than it looks. A pure-search investigation has a
failure mode where the top hit is a plausible neighbour — search "brute force
authentication" and ``T1187 Forced Authentication`` comes back second — and a
narrative built on it is fluent, cited, and about the wrong technique. Citations
make hallucination visible only when the thing being cited is checkable, and an
identifier lookup is checkable in a way that a relevance score is not.

Three sources of evidence, all resolvable
-----------------------------------------
``kb_chunk`` / ``attack_technique`` / ``cve_record``
    From the knowledge base, refs resolvable via
    :meth:`~sentinel.kb.retrieve.KnowledgeBase.resolve`.
``alert_field``
    A raw log line, in F-05's own words. Ref is ``alert://<id>#<field>`` and the
    excerpt is typed untrusted, because an attacker chose the value.
``model_output``
    The detector's score and the threshold it was compared against. Cited
    because "the ensemble scored 0.97" is a factual claim like any other, and an
    uncited model score in an analyst-facing narrative is precisely the
    unexplained black-box output PRD Section 9.2 forbids in the demo.

Knowledge-base poisoning
------------------------
The knowledge base is built from public corpora, so a poisoned entry is a
supply-chain attack on the investigation itself — and the default corpus ships
one, deliberately, so this path is exercised rather than assumed.
:meth:`~sentinel.kb.retrieve.KnowledgeBase.search` excludes chunks that scan as
instruction-like, and :func:`build_report` additionally refuses to cite one
reached by lookup, since a lookup bypasses the search filter. The poisoning is
reported as a finding rather than silently dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from sentinel.agents.engine import (
    NullEngine,
    ReasoningEngine,
    parse_narrative_opinion,
    reconcile_claims,
    sanitize_engine_text,
)
from sentinel.agents.prompts import INVESTIGATION_SYSTEM_PROMPT, AgentPrompt
from sentinel.agents.triage import BENIGN_FAMILY, TECHNIQUE_BY_FAMILY
from sentinel.core.clock import Clock, SystemClock
from sentinel.core.errors import SentinelError
from sentinel.core.ids import deterministic_id
from sentinel.core.schemas import (
    ActionType,
    Alert,
    Evidence,
    EvidenceKind,
    InvestigationReport,
    Severity,
    TriageResult,
)
from sentinel.kb.corpus import DocumentKind
from sentinel.kb.retrieve import KnowledgeBase

__all__ = [
    "LINKED_RELEVANCE",
    "RECOMMENDED_ACTIONS",
    "InvestigationAgent",
    "InvestigationError",
    "alert_field_evidence",
    "model_output_evidence",
]


class InvestigationError(SentinelError):
    """The Investigation Agent could not build a grounded report."""


RECOMMENDED_ACTIONS: Final[dict[str, tuple[ActionType, ...]]] = {
    "brute_force": (ActionType.BLOCK_IP, ActionType.DISABLE_ACCOUNT),
    "web_attack": (ActionType.BLOCK_IP, ActionType.OPEN_PATCH_PR),
    "recon": (ActionType.BLOCK_IP,),
    "dos": (ActionType.BLOCK_IP,),
    "ddos": (ActionType.BLOCK_IP,),
    "botnet": (ActionType.ISOLATE_HOST, ActionType.BLOCK_IP),
    "infiltration": (ActionType.ISOLATE_HOST, ActionType.QUARANTINE_FILE),
}
"""Family → the actions an analyst would consider. Advisory only.

The Containment Agent proposes, the trust tier decides whether a human must
approve, and :meth:`~sentinel.core.schemas.ActionRequest.propose` derives that
flag. Nothing here can authorise anything; this is the investigation's opinion
about what the options are.
"""

#: Alert attributes worth citing as raw-log evidence. Chosen to be the fields an
#: analyst would read out loud when describing the incident.
_CITABLE_FIELDS: Final[tuple[str, ...]] = (
    "src_ip",
    "dst_ip",
    "dst_port",
    "protocol",
    "asset_id",
    "signature",
)


def alert_field_evidence(alert: Alert, field_name: str) -> Evidence | None:
    """Cite one alert field as a raw log line. ``None`` if the field is absent."""
    value = getattr(alert, field_name, None)
    if value is None:
        value = alert.features.get(field_name)
    if value is None:
        return None
    return Evidence(
        kind=EvidenceKind.ALERT_FIELD,
        ref=f"alert://{alert.alert_id}#{field_name}",
        # Untrusted by construction: an attacker chose the source address, the
        # port and, on a signature-bearing feed, much of the signature text.
        excerpt=f"{field_name}={value}",
        relevance=1.0,
    )


def model_output_evidence(triage: TriageResult) -> Evidence:
    """Cite the detector score behind the verdict."""
    return Evidence(
        kind=EvidenceKind.MODEL_OUTPUT,
        ref=f"model://{triage.model_version}#anomaly_score",
        excerpt=(
            f"anomaly_score={triage.anomaly_score if triage.anomaly_score is not None else 'n/a'}"
            f" confidence={triage.confidence:.3f} decision={triage.decision.value}"
        ),
        relevance=1.0,
    )


@dataclass(slots=True)
class InvestigationAgent:
    """Builds a cited, MITRE-mapped narrative for an escalated alert."""

    kb: KnowledgeBase
    engine: ReasoningEngine = field(default_factory=NullEngine)
    clock: Clock = field(default_factory=SystemClock)
    #: How many chunks the contextual search may contribute.
    search_k: int = 4
    version: str = "investigation-1"

    def investigate(
        self, alert: Alert, *, now: datetime | None = None
    ) -> InvestigationReport:
        """Produce a grounded report. Raises if the alert has not been triaged."""
        triage = alert.triage
        if triage is None:
            raise InvestigationError(
                f"alert {alert.alert_id} has no triage verdict; the Investigation Agent "
                "runs after triage and needs its technique mapping to look anything up"
            )
        created_at = now or self.clock.now()

        evidence, findings, looked_up = self._gather(alert, triage)
        claims = self._claims(alert, triage, evidence, findings)

        if not isinstance(self.engine, NullEngine):
            claims, summary = self._with_engine(alert, triage, evidence, claims)
        else:
            summary = self._summary(alert, triage, findings)

        techniques = self._techniques(triage, looked_up)
        return InvestigationReport(
            report_id=deterministic_id("investigation", alert.alert_id, self.version),
            alert_ids=(alert.alert_id,),
            tenant_id=alert.tenant_id,
            summary=summary,
            claims=claims,
            evidence=evidence,
            techniques=techniques,
            severity=triage.severity,
            confidence=triage.confidence,
            recommended_actions=self._recommended(triage),
            model_version=self.version,
            created_at=created_at,
        )

    # --- evidence ------------------------------------------------------------- #

    def _gather(
        self, alert: Alert, triage: TriageResult
    ) -> tuple[tuple[Evidence, ...], list[str], tuple[str, ...]]:
        """Collect every citation, deduplicated by ref.

        Returns the evidence, any security findings raised while collecting it,
        and the technique ids reached *by lookup* — which is the only provenance
        the report is willing to assert a technique mapping from.
        """
        collected: dict[str, Evidence] = {}
        findings: list[str] = []
        looked_up: list[str] = []
        retrieved_at = self.clock.now()

        def add(item: Evidence | None) -> None:
            if item is not None and item.ref not in collected:
                collected[item.ref] = item

        def cite(doc_id: str, *, kind: EvidenceKind, relevance: float, limit: int) -> int:
            """Cite up to ``limit`` chunks of one document, skipping poisoned ones."""
            cited = 0
            for chunk in self.kb.chunks_for(doc_id):
                if cited >= limit:
                    break
                scan = self._scan_for(chunk.chunk_id)
                if scan is not None and scan.is_attack_indicator:
                    # A lookup bypasses search()'s filter, so it is re-applied here.
                    findings.append(
                        f"knowledge-base chunk {chunk.chunk_id} reads as an instruction "
                        f"({scan.summary()}); excluded from citation and reported"
                    )
                    continue
                add(
                    chunk.as_evidence(
                        evidence_kind=kind, relevance=relevance, retrieved_at=retrieved_at
                    )
                )
                cited += 1
            return cited

        if triage.technique_id is not None:
            # 1. The identifier triage already established.
            if cite(
                triage.technique_id,
                kind=EvidenceKind.ATTACK_TECHNIQUE,
                relevance=1.0,
                limit=2,
            ):
                looked_up.append(triage.technique_id)
            # 2. What the relation graph says it connects to, partitioned by kind
            #    so one technique's thirty linked CVEs cannot crowd out its one
            #    playbook — which is the citation the analyst actually opens.
            budget = dict(_RELATED_BUDGET)
            for related_id in self.kb.related(triage.technique_id):
                chunks = self.kb.chunks_for(related_id)
                if not chunks:
                    continue
                doc_kind = chunks[0].kind
                if budget.get(doc_kind, 0) <= 0:
                    continue
                if cite(
                    related_id,
                    kind=_EVIDENCE_KIND_FOR_DOC.get(doc_kind, EvidenceKind.KB_CHUNK),
                    relevance=LINKED_RELEVANCE,
                    limit=1,
                ):
                    budget[doc_kind] -= 1
                    if doc_kind == DocumentKind.TECHNIQUE.value:
                        looked_up.append(related_id)
        else:
            # 3. No identifier to look up — a novelty-only escalation, where the
            #    detector fired and the classifier named no family. Free-text
            #    search is the only option, and it is used *only* here. Measured
            #    on this corpus, adding it alongside a successful lookup returns
            #    plausible neighbours ranked above the right answer (a DDoS query
            #    surfaces T1110 and T1046 detection text), and the retriever's
            #    relevance score does not separate them — the correct hit scored
            #    0.025 while a wrong neighbour scored 0.066 — so there is no
            #    threshold that would filter them. Citing by identifier when an
            #    identifier exists, and only falling back to ranking when one does
            #    not, is the version of this that cannot quietly be wrong.
            for hit in self.kb.search(
                self._query(alert, triage), k=self.search_k, retrieved_at=retrieved_at
            ):
                add(hit.as_evidence(retrieved_at=retrieved_at))

        # 4. The raw log lines and the model output.
        for name in _CITABLE_FIELDS:
            add(alert_field_evidence(alert, name))
        add(model_output_evidence(triage))

        if alert.injection_scan.is_attack_indicator:
            findings.append(
                f"alert payload reads as an instruction to the agents "
                f"({alert.injection_scan.summary()})"
            )
        return tuple(collected.values()), findings, tuple(dict.fromkeys(looked_up))

    def _scan_for(self, chunk_id: str):
        for chunk, scan in zip(self.kb.chunks, self.kb.scans, strict=True):
            if chunk.chunk_id == chunk_id:
                return scan
        return None

    def _query(self, alert: Alert, triage: TriageResult) -> str:
        """The search text. Built from typed fields, never from the raw payload.

        Retrieval over attacker-controlled text is a ranking attack: a payload
        containing "ransomware encryption T1486" steers the investigation's
        evidence without ever having to defeat the prompt fence. The query is
        therefore assembled from the fields the *system* derived — the technique,
        the severity, the port and protocol — and the payload contributes
        nothing to it.
        """
        parts: list[str] = []
        if triage.technique_id is not None:
            parts.append(triage.technique_id)
        parts.append(triage.severity.value)
        if alert.dst_port is not None:
            parts.append(f"port {alert.dst_port}")
        if alert.protocol:
            parts.append(alert.protocol)
        for name in triage.supporting_fields:
            parts.append(name.replace("_", " "))
        return " ".join(parts) or "security incident response"

    # --- claims --------------------------------------------------------------- #

    def _claims(
        self,
        alert: Alert,
        triage: TriageResult,
        evidence: Sequence[Evidence],
        findings: Sequence[str],
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Build the deterministic claims. Every one cites before it asserts."""
        by_kind: dict[EvidenceKind, list[str]] = {}
        for item in evidence:
            by_kind.setdefault(item.kind, []).append(item.ref)

        claims: list[tuple[str, tuple[str, ...]]] = []
        model_refs = tuple(by_kind.get(EvidenceKind.MODEL_OUTPUT, ()))
        field_refs = tuple(by_kind.get(EvidenceKind.ALERT_FIELD, ()))
        technique_refs = tuple(by_kind.get(EvidenceKind.ATTACK_TECHNIQUE, ()))
        cve_refs = tuple(by_kind.get(EvidenceKind.CVE_RECORD, ()))
        playbook_refs = tuple(by_kind.get(EvidenceKind.KB_CHUNK, ()))

        if model_refs:
            score = (
                f"{triage.anomaly_score:.3f}" if triage.anomaly_score is not None else "n/a"
            )
            claims.append(
                (
                    f"The detection ensemble scored this flow {score} and triage "
                    f"decided {triage.decision.value} at confidence "
                    f"{triage.confidence:.2f}.",
                    model_refs,
                )
            )
        if field_refs:
            claims.append(
                (
                    f"The activity involves asset {alert.asset_id} and is recorded on the "
                    f"{alert.source.value} feed.",
                    field_refs,
                )
            )
        if triage.technique_id is not None and technique_refs:
            cited = tuple(
                ref
                for ref in field_refs
                if ref.rsplit("#", 1)[-1] in triage.supporting_fields
            )
            claims.append(
                (
                    f"The observed pattern maps to MITRE ATT&CK {triage.technique_id}, "
                    f"supported by {', '.join(triage.supporting_fields) or 'detector output'}.",
                    technique_refs + (cited or model_refs),
                )
            )
        if cve_refs:
            claims.append(
                (
                    "The knowledge base links this technique to known vulnerabilities, "
                    "which are the concrete ways it is realised in the wild.",
                    cve_refs,
                )
            )
        if playbook_refs:
            claims.append(
                (
                    "Response guidance for this technique is available in the indexed "
                    "playbooks and advisories.",
                    playbook_refs,
                )
            )
        for finding in findings:
            # A finding about the *input* is grounded in the input, so it cites the
            # alert fields rather than the knowledge base.
            if field_refs or model_refs:
                claims.append((finding, field_refs or model_refs))
        return tuple(claims)

    def _summary(
        self, alert: Alert, triage: TriageResult, findings: Sequence[str]
    ) -> str:
        head = (
            f"{triage.severity.value.upper()} — {triage.decision.value} on asset "
            f"{alert.asset_id}"
        )
        if triage.technique_id is not None:
            head += f" mapped to {triage.technique_id}"
        body = triage.rationale
        tail = f" {len(findings)} security finding(s) raised." if findings else ""
        return f"{head}. {body}{tail}"[:8000]

    def _techniques(
        self, triage: TriageResult, looked_up: Sequence[str]
    ) -> tuple[str, ...]:
        """Techniques the report asserts — lookup provenance only.

        A technique reached by text search is a *neighbour*, not a mapping. The
        search that finds ``T1110 Brute Force`` for a DDoS alert finds it because
        both descriptions talk about repeated connections, and a report asserting
        that mapping is fluent, cited and wrong. Only the identifier triage
        established, and the sub-techniques the relation graph links it to, are
        asserted here.
        """
        cited: list[str] = []
        if triage.technique_id is not None:
            cited.append(triage.technique_id)
        cited.extend(looked_up)
        return tuple(dict.fromkeys(cited))

    def _recommended(self, triage: TriageResult) -> tuple[ActionType, ...]:
        for family, technique in TECHNIQUE_BY_FAMILY.items():
            if technique == triage.technique_id and family != BENIGN_FAMILY:
                return RECOMMENDED_ACTIONS.get(family, (ActionType.NOTIFY_ANALYST,))
        return (
            (ActionType.NOTIFY_ANALYST,)
            if triage.severity >= Severity.HIGH
            else (ActionType.ENRICH_ONLY,)
        )

    # --- engine --------------------------------------------------------------- #

    def _with_engine(
        self,
        alert: Alert,
        triage: TriageResult,
        evidence: Sequence[Evidence],
        deterministic: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], str]:
        """Ask for a narrative, keep only what it can cite.

        The deterministic claims are never replaced, only appended to. A model
        that declines, times out, or hallucinates every citation leaves the
        report exactly as it was — which is the property that lets the engine be
        optional rather than load-bearing.
        """
        response = self.engine.respond(self._prompt(alert, triage, evidence))
        opinion = parse_narrative_opinion(response)
        if opinion is None:
            return deterministic, self._summary(alert, triage, ())

        resolvable = [item.ref for item in evidence]
        kept = reconcile_claims(opinion.claims, resolvable=resolvable)
        existing = {statement for statement, _refs in deterministic}
        merged = deterministic + tuple(
            (statement, refs) for statement, refs in kept if statement not in existing
        )
        summary = self._summary(alert, triage, ())
        if opinion.summary:
            addition = sanitize_engine_text(opinion.summary, engine=response.engine)
            if addition:
                summary = f"{summary} {addition}"[:8000]
        return merged, summary

    def _prompt(
        self, alert: Alert, triage: TriageResult, evidence: Sequence[Evidence]
    ) -> AgentPrompt:
        catalogue = "\n".join(
            f"  {item.ref}  [{item.kind.value}]" for item in evidence
        )
        task = (
            f"Alert {alert.alert_id} on asset {alert.asset_id} was triaged "
            f"{triage.decision.value} at severity {triage.severity.value}"
            + (
                f", mapped to {triage.technique_id}"
                if triage.technique_id is not None
                else ""
            )
            + ".\n\nEvidence refs you may cite, and no others:\n"
            + catalogue
            + "\n\nWrite the narrative. Every claim cites at least one ref above."
        )
        schema = (
            '{"summary": "string", "claims": [["statement", ["ref"]]], '
            '"techniques": ["T####"], "poisoning_detected": true|false}'
        )
        prompt = AgentPrompt(
            system=INVESTIGATION_SYSTEM_PROMPT, task=task, response_schema=schema
        )
        for item in evidence:
            if item.kind in (EvidenceKind.ALERT_FIELD, EvidenceKind.RAW_LOG):
                prompt = prompt.with_untrusted(
                    item.excerpt, label="alert.field", origin="alert.field"
                )
        return prompt


_EVIDENCE_KIND_FOR_DOC: Final[dict[str, EvidenceKind]] = {
    kind.value: kind.evidence_kind for kind in DocumentKind
}

#: How many linked documents of each kind may be cited. Budgeted per kind
#: because ``T1190`` links to thirty-one CVEs and two playbooks; a flat "first
#: three links" rule would cite three CVEs and never reach the playbook, which
#: is the citation an analyst responding to the incident actually needs.
_RELATED_BUDGET: Final[dict[str, int]] = {
    DocumentKind.PLAYBOOK.value: 2,
    DocumentKind.CVE.value: 2,
    DocumentKind.TECHNIQUE.value: 2,
    DocumentKind.ADVISORY.value: 1,
}

#: Relevance recorded for a structurally linked citation. Below 1.0 because the
#: link is a fact about the corpus rather than a match against this alert, and
#: mirroring how ``graph/explain.py`` prices an inherited contribution below an
#: intrinsic one.
LINKED_RELEVANCE: Final[float] = 0.8
