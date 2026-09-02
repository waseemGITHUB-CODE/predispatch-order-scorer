"""Readable labels for the Brazilian states in the data.

The data is Brazilian and the model is trained on Brazilian state codes. `SP`
and `BA` mean nothing to a reader who has not worked with it, so the UI shows

    (SP) São Paulo — Maharashtra

carrying the code, the real state, and the Indian state that plays the nearest
equivalent role. Three things about that third column matter:

**It is display only.** `customer_state` and `seller_state` are ordinal-encoded
with `unknown_value=-1`. An Indian name posted to the model would encode as
unknown, collapsing both features to a constant and silently costing about 18%
of the model's entire edge over chance. Everything here changes what is *shown*;
the value sent to the model is always the Brazilian code.

**The basis is role in the delivery network**, not any claim that the two places
have comparable failure rates, distances or volumes. São Paulo is 46% of buyers
and 70% of sellers — it is the fulfilment hub, so it maps to Maharashtra. From
there the ordering is: adjacent developed, southern developed, capital district,
distant north-east, remote frontier.

**Beyond the top twelve the analogy is loose.** Those twelve are 94% of orders
and the pairings are arguable. The remaining fifteen exist so the list is
consistent rather than half-labelled; they carry no analytical weight and the
README says so.
"""
from __future__ import annotations

# code: (Brazilian state, Indian state playing the nearest role)
# Ordered by share of orders, so the dropdown opens on the ones that matter.
STATES: dict[str, tuple[str, str]] = {
    # the hub and the states around it
    "SP": ("São Paulo", "Maharashtra"),          # 46% of buyers, 70% of sellers
    "RJ": ("Rio de Janeiro", "Delhi NCR"),       # second metro
    "MG": ("Minas Gerais", "Gujarat"),           # large adjacent, efficient
    "PR": ("Paraná", "Karnataka"),               # southern industrial
    "RS": ("Rio Grande do Sul", "Tamil Nadu"),   # far south, developed
    "BA": ("Bahia", "Bihar"),                    # north-east, distant, worst failure rate
    "SC": ("Santa Catarina", "Kerala"),          # small southern, efficient
    "DF": ("Distrito Federal", "Chandigarh"),    # federal capital district
    "ES": ("Espírito Santo", "Goa"),             # small coastal
    "GO": ("Goiás", "Madhya Pradesh"),           # central, agricultural
    "PE": ("Pernambuco", "West Bengal"),         # north-east coastal
    "CE": ("Ceará", "Odisha"),                   # north-east, distant
    # --- below here the analogy is loose, ordered by remoteness ---
    "PA": ("Pará", "Assam"),
    "MT": ("Mato Grosso", "Rajasthan"),
    "MA": ("Maranhão", "Jharkhand"),
    "MS": ("Mato Grosso do Sul", "Haryana"),
    "PB": ("Paraíba", "Chhattisgarh"),
    "RN": ("Rio Grande do Norte", "Uttarakhand"),
    "PI": ("Piauí", "Tripura"),
    "AL": ("Alagoas", "Himachal Pradesh"),
    "SE": ("Sergipe", "Meghalaya"),
    "AM": ("Amazonas", "Arunachal Pradesh"),
    "RO": ("Rondônia", "Nagaland"),
    "TO": ("Tocantins", "Manipur"),
    "AC": ("Acre", "Mizoram"),
    "AP": ("Amapá", "Sikkim"),
    "RR": ("Roraima", "Ladakh"),
}

# The first twelve carry real weight; the rest are there for a consistent list.
MEANINGFUL = 12


def label(code: str) -> str:
    """`(SP) São Paulo — Maharashtra`, or the bare code if unrecognised."""
    pair = STATES.get(code)
    return f"({code}) {pair[0]} — {pair[1]}" if pair else code


def short(code: str) -> str:
    """`Bahia (BA)` — for prose, where the full triple would crowd the sentence."""
    pair = STATES.get(code)
    return f"{pair[0]} ({code})" if pair else code


def as_options() -> list[dict]:
    """The dropdown payload, in the order the UI should render it."""
    return [
        {
            "code": code,
            "brazilian": brazilian,
            "indian": indian,
            "label": label(code),
            "meaningful": i < MEANINGFUL,
        }
        for i, (code, (brazilian, indian)) in enumerate(STATES.items())
    ]
