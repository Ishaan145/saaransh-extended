"""
saaransh-unified.case_studies.case_a_finfluencer
================================================
Case A from paper §6: SEBI 2024 discussion paper on the regulation of
unregistered financial influencers ('finfluencers').

This module assembles the inputs for the reconstruction pipeline:
* policy template (Sec. 6.1, Case A) used as the substrate-only prior
  when engine outputs are unavailable;
* synthetic comment stream calibrated to the documented participation
  skew (urban, English-literate, male, tertiary-educated, higher
  income, general-category over-representation).

The comment text is templated from realistic stakeholder-voice
sentences; the labels they will receive from the sentiment ensemble
are not pre-set but emerge from the polarity of the templated text.
This keeps the evaluation honest: the NLP layer scores comments it has
not seen before, and the demographic labels are independent metadata.
"""

from __future__ import annotations
import random


POLICY = {
    "name": "SEBI 2024 finfluencer regulation",
    "policy_id": "SEBI_FINFLUENCER_2024",
    "domain": "finance",
    "summary": (
        "Restrict unregistered individuals from providing securities "
        "recommendations on social media outside the registered "
        "investment-adviser framework."
    ),
    # Substrate-only prior weights. Magnitudes are deliberately modest:
    # the substrate path is a fallback, not the headline path.
    "template_weights": {
        # Investors in higher income deciles benefit from formalised
        # advisory standards (positive on net welfare).
        "income:d9_10":     0.35,
        "income:d6_8":      0.20,
        # First-time low-income investors in T2/T3 lose access to free
        # informal advice channels (negative on net welfare).
        "income:d1_3":      -0.30,
        "income:d4_5":      -0.20,
        "tier:tier2":       -0.15,
        "tier:rural":       -0.20,
        "tier:metro":        0.15,
        # Tertiary and postgrad audiences are net supportive: better
        # access to registered advisers, less reliant on finfluencers.
        "edu:tertiary":      0.20,
        "edu:postgrad":      0.30,
        "edu:below_secondary": -0.20,
        # Age: older audiences support stronger investor protection;
        # younger 18-25 are heavy finfluencer consumers and net oppose.
        "age:60p":           0.25,
        "age:41_60":         0.15,
        "age:18_25":        -0.25,
        "age:26_40":        -0.05,
    },
    "template_bias": -0.05,
}


# ------------------------------------------------------------ templates
# Each template carries an implicit polarity but the sentiment ensemble
# scores them on its own.
POSITIVE_TEMPLATES = [
    "I strongly support SEBI's proposal to regulate finfluencers. Retail investors deserve protection from unregistered advice.",
    "This regulation is necessary and beneficial. Unregistered influencers have caused real harm to small investors.",
    "I endorse the move. Formal accountability for financial recommendations is overdue and welcome.",
    "Excellent step. The current ecosystem allows clearly unfair practices that hurt new investors.",
    "I welcome this regulation. Investor protection is essential and SEBI is right to step in.",
    "Strongly approve. Bringing influencers under the adviser framework will improve market integrity.",
    "This is a positive and progressive regulation. It addresses a real harm and is justified.",
    "I agree with the proposed framework. Registration requirements are fair and protect retail savers.",
]

NEGATIVE_TEMPLATES = [
    "I oppose this regulation. It will silence small creators who genuinely educate first-time investors.",
    "This is regressive and unfair to retail commentators. Free speech on personal finance must not be restricted.",
    "Strongly disagree. The framework is overreaching and burdensome for individual creators.",
    "This is bad policy. It privileges large registered firms and harms independent voices.",
    "The proposal is problematic and excludes ordinary investors who learn from peer creators.",
    "I disapprove of the draft. The compliance burden is disproportionate and concerning.",
    "Reject this regulation. It is ambiguous and will discriminate against small creators.",
    "This is harmful and unnecessary. Self-regulation already addresses most of the issues raised.",
]

NEUTRAL_TEMPLATES = [
    "The regulation has both useful elements and concerning provisions. The disclosure rules are reasonable but the registration scope is unclear.",
    "Some parts of the proposal are sensible while others may need refinement. The thresholds should be clarified.",
    "I have mixed views. The intent is good, the implementation is partially problematic.",
    "There are valid arguments on both sides. The draft needs more clarity on enforcement.",
    "The proposal is somewhat ambiguous. It is partly necessary but partly an overreach.",
]


def _draw_demographic(skew: str, rng: random.Random) -> dict:
    """Sample a demographic record. `skew` is 'participant' (the heavily
    self-selected online sample) or 'population' (eligible adults).
    """
    if skew == "participant":
        return {
            "age":      rng.choices(("18_25","26_40","41_60","60p"),    weights=[0.30,0.50,0.18,0.02])[0],
            "sex":      rng.choices(("M","F"),                          weights=[0.78,0.22])[0],
            "tier":     rng.choices(("metro","tier2","rural"),          weights=[0.62,0.30,0.08])[0],
            "region":   rng.choices(("hindi_belt","south","east","west","northeast","uts"),
                                    weights=[0.30,0.28,0.10,0.25,0.02,0.05])[0],
            "education":rng.choices(("below_secondary","secondary","tertiary","postgrad"),
                                    weights=[0.03,0.15,0.55,0.27])[0],
            "income":   rng.choices(("d1_3","d4_5","d6_8","d9_10"),     weights=[0.08,0.17,0.40,0.35])[0],
            "caste":    rng.choices(("general","obc","sc_st","minority"),weights=[0.55,0.30,0.10,0.05])[0],
        }
    # Approximate eligible-adult marginals (matches ELIGIBLE_POP_MARGINALS).
    return {
        "age":      rng.choices(("18_25","26_40","41_60","60p"),    weights=[0.20,0.36,0.30,0.14])[0],
        "sex":      rng.choices(("M","F"),                          weights=[0.512,0.488])[0],
        "tier":     rng.choices(("metro","tier2","rural"),          weights=[0.18,0.32,0.50])[0],
        "region":   rng.choices(("hindi_belt","south","east","west","northeast","uts"),
                                weights=[0.43,0.21,0.16,0.13,0.04,0.03])[0],
        "education":rng.choices(("below_secondary","secondary","tertiary","postgrad"),
                                weights=[0.42,0.34,0.18,0.06])[0],
        "income":   rng.choices(("d1_3","d4_5","d6_8","d9_10"),     weights=[0.30,0.20,0.30,0.20])[0],
        "caste":    rng.choices(("general","obc","sc_st","minority"),weights=[0.28,0.42,0.245,0.055])[0],
    }


def _draw_opinion_text(demographic: dict, rng: random.Random) -> str:
    """Conditional comment generator.

    A demographic-conditional probability of supporting / opposing /
    being neutral on the policy is drawn, then a template is sampled
    from the corresponding bucket. The conditional probabilities reflect
    the substrate prior (`POLICY.template_weights`) so that the
    sentiment ensemble, demographic data and synthetic prior all
    correlate in the expected direction.
    """
    # Score for the demographic from the policy template.
    score = 0.0
    for k, v in POLICY["template_weights"].items():
        ax, val = k.split(":", 1)
        if demographic.get(ax) == val:
            score += v
    score += POLICY["template_bias"]
    # Map to a probability triple (support, oppose, neutral).
    base_p = max(0.05, min(0.85, 0.5 + 0.6 * score))   # p(support)
    base_n = max(0.05, min(0.85, 0.5 - 0.6 * score))   # p(oppose)
    base_t = max(0.05, 1.0 - base_p - base_n)
    z = base_p + base_n + base_t
    p_s, p_o, p_t = base_p/z, base_n/z, base_t/z
    label = rng.choices(("support","oppose","neutral"), weights=[p_s, p_o, p_t])[0]
    if label == "support":
        return rng.choice(POSITIVE_TEMPLATES)
    if label == "oppose":
        return rng.choice(NEGATIVE_TEMPLATES)
    return rng.choice(NEUTRAL_TEMPLATES)


def generate_comments(n: int, seed: int = 20260601) -> list[dict]:
    """Produce a synthetic comment stream calibrated to the documented
    participation skew of SEBI's discussion-paper consultations.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d = _draw_demographic("participant", rng)
        txt = _draw_opinion_text(d, rng)
        out.append({"text": txt, "demographic": d})
    return out
