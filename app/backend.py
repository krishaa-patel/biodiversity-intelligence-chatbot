import os
import re
import json
import time
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# PROJECT PATHS + API
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "chroma_db"

load_dotenv(ROOT_DIR / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Check the .env file "
        "in the main project folder."
    )


# =========================================================
# EMBEDDINGS + VECTOR DATABASE
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="biodiversity_knowledge"
)


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================================================
# RATE LIMIT RETRY
# =========================================================

def invoke_with_retry(runnable, prompt, max_retries=6):

    for attempt in range(max_retries):

        try:
            return runnable.invoke(prompt)

        except Exception as e:

            error_text = str(e).lower()

            is_rate_limit = (
                "429" in error_text
                or "rate_limit" in error_text
                or "rate limit" in error_text
            )

            if not is_rate_limit:
                raise

            seconds_match = re.search(
                r"try again in\s+([\d.]+)s",
                error_text
            )

            if seconds_match:
                wait_time = float(seconds_match.group(1)) + 1
            else:
                wait_time = min(
                    4 * (attempt + 1),
                    20
                )

            print(
                f"Groq rate limit reached. "
                f"Retrying in {wait_time:.1f}s..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Groq remained rate-limited after several retries."
    )


# =========================================================
# DATA MODELS
# =========================================================

class EnvironmentalProfile(BaseModel):

    soil_organic_carbon: Optional[float] = None
    soil_ph: Optional[float] = None
    soil_moisture: Optional[str] = None

    rainfall: Optional[str] = None
    temperature: Optional[str] = None

    land_use: Optional[str] = None
    crop: Optional[str] = None

    biodiversity_trend: Optional[str] = None
    species_richness: Optional[str] = None
    habitat_diversity: Optional[str] = None

    pollution: Optional[str] = None
    deforestation: Optional[str] = None

    region: Optional[str] = None


class GroundingCheck(BaseModel):

    supported_claims: List[str]
    unsupported_claims: List[str]
    explanation: str


grounding_llm = llm.with_structured_output(
    GroundingCheck,
    method="json_schema"
)


# =========================================================
# DETERMINISTIC PROFILE EXTRACTION
# =========================================================

def extract_environmental_profile(user_text):

    text = user_text.lower().strip()

    data = {}

    # -----------------------------------------------------
    # SOIL ORGANIC CARBON
    # -----------------------------------------------------

    soc_match = re.search(
        r"(?:soil organic carbon|soil carbon|\bsoc\b)"
        r"[^\d]{0,30}(\d+(?:\.\d+)?)\s*%?",
        text
    )

    if soc_match:
        data["soil_organic_carbon"] = float(
            soc_match.group(1)
        )

    # -----------------------------------------------------
    # SOIL PH
    # -----------------------------------------------------

    ph_match = re.search(
        r"(?:soil\s+)?ph[^\d]{0,15}"
        r"(\d+(?:\.\d+)?)",
        text
    )

    if ph_match:
        data["soil_ph"] = float(
            ph_match.group(1)
        )

    # -----------------------------------------------------
    # RAINFALL
    # -----------------------------------------------------

    rainfall_number = re.search(
        r"(?:rainfall|rain)[^\d]{0,20}"
        r"(\d+(?:\.\d+)?)\s*"
        r"(mm|millimeters?|inches?|inch)",
        text
    )

    if rainfall_number:

        data["rainfall"] = (
            rainfall_number.group(1)
            + " "
            + rainfall_number.group(2)
        )

    elif (
        "low rainfall" in text
        or re.search(
            r"(?:rainfall|rain).{0,20}\blow\b",
            text
        )
    ):
        data["rainfall"] = "low"

    elif (
        "high rainfall" in text
        or re.search(
            r"(?:rainfall|rain).{0,20}\bhigh\b",
            text
        )
    ):
        data["rainfall"] = "high"

    elif (
        "moderate rainfall" in text
        or re.search(
            r"(?:rainfall|rain).{0,20}\bmoderate\b",
            text
        )
    ):
        data["rainfall"] = "moderate"

    # -----------------------------------------------------
    # LAND USE
    # -----------------------------------------------------

    agriculture_terms = [
        "farm",
        "farmland",
        "crop field",
        "cropland",
        "agricultural land",
        "agriculture"
    ]

    if any(
        term in text
        for term in agriculture_terms
    ):
        data["land_use"] = "agriculture"

    elif "forest" in text:
        data["land_use"] = "forest"

    elif any(
        term in text
        for term in [
            "pasture",
            "grazing land",
            "grassland"
        ]
    ):
        data["land_use"] = "pasture"

    # -----------------------------------------------------
    # CROP
    # -----------------------------------------------------

    crops = [
        "wheat",
        "rice",
        "maize",
        "corn",
        "soybean",
        "soy",
        "cotton",
        "barley",
        "millet",
        "sorghum"
    ]

    for crop in crops:

        if crop in text:

            if crop == "corn":
                data["crop"] = "maize"

            elif crop == "soy":
                data["crop"] = "soybean"

            else:
                data["crop"] = crop

            break

    # -----------------------------------------------------
    # BIODIVERSITY TREND
    # -----------------------------------------------------

    declining_phrases = [
        "biodiversity is declining",
        "biodiversity is decreasing",
        "biodiversity has declined",
        "biodiversity has been declining",
        "biodiversity decline"
    ]

    if any(
        phrase in text
        for phrase in declining_phrases
    ):
        data["biodiversity_trend"] = "declining"

    elif (
        "biodiversity is increasing" in text
        or "biodiversity is improving" in text
    ):
        data["biodiversity_trend"] = "improving"

    # -----------------------------------------------------
    # REGION / CLIMATE
    # -----------------------------------------------------

    if (
        "semi-arid" in text
        or "semi arid" in text
    ):
        data["region"] = "semi-arid"

    elif re.search(r"\barid\b", text):
        data["region"] = "arid"

    elif "tropical" in text:
        data["region"] = "tropical"

    elif "temperate" in text:
        data["region"] = "temperate"

    elif "mediterranean" in text:
        data["region"] = "mediterranean"

    elif "humid" in text:
        data["region"] = "humid"

    # -----------------------------------------------------
    # SOIL MOISTURE
    # -----------------------------------------------------

    if "soil moisture is low" in text:
        data["soil_moisture"] = "low"

    elif "soil moisture is high" in text:
        data["soil_moisture"] = "high"

    elif "soil moisture is moderate" in text:
        data["soil_moisture"] = "moderate"

    # -----------------------------------------------------
    # SPECIES RICHNESS
    # -----------------------------------------------------

    if "species richness is low" in text:
        data["species_richness"] = "low"

    elif "species richness is high" in text:
        data["species_richness"] = "high"

    elif "species richness is declining" in text:
        data["species_richness"] = "declining"

    # -----------------------------------------------------
    # HABITAT DIVERSITY
    # -----------------------------------------------------

    if "habitat diversity is low" in text:
        data["habitat_diversity"] = "low"

    elif "habitat diversity is high" in text:
        data["habitat_diversity"] = "high"

    elif "habitat diversity is declining" in text:
        data["habitat_diversity"] = "declining"

    # -----------------------------------------------------
    # POLLUTION
    # -----------------------------------------------------

    if "pollution is increasing" in text:
        data["pollution"] = "increasing"

    elif "high pollution" in text:
        data["pollution"] = "high"

    elif "low pollution" in text:
        data["pollution"] = "low"

    # -----------------------------------------------------
    # DEFORESTATION
    # -----------------------------------------------------

    if "deforestation is increasing" in text:
        data["deforestation"] = "increasing"

    elif "high deforestation" in text:
        data["deforestation"] = "high"

    elif "deforestation" in text:
        data["deforestation"] = "present"

    return EnvironmentalProfile(**data)


# =========================================================
# MERGE PROFILE MEMORY
# =========================================================

def merge_profiles(old_profile, new_profile):

    old_data = old_profile.model_dump()
    new_data = new_profile.model_dump()

    for key, value in new_data.items():

        if value is not None:
            old_data[key] = value

    return EnvironmentalProfile(
        **old_data
    )


# =========================================================
# REQUIRED INFORMATION
# =========================================================

IMPORTANT_FIELDS = {

    "soil_organic_carbon":
        "soil organic carbon percentage",

    "rainfall":
        "rainfall level or annual rainfall",

    "land_use":
        "current land use",

    "region":
        "region or climate type"
}


def find_missing_fields(profile):

    missing = []

    data = profile.model_dump()

    for field, description in (
        IMPORTANT_FIELDS.items()
    ):

        if data.get(field) is None:

            missing.append(
                (field, description)
            )

    return missing


# =========================================================
# DETERMINISTIC CLARIFYING QUESTION
# =========================================================

def generate_clarifying_question(profile):

    missing = find_missing_fields(profile)

    if not missing:
        return None

    lines = [
        "Could you please provide the following missing information?"
    ]

    for _, description in missing:

        lines.append(
            f"- {description.capitalize()}"
        )

    lines.append(
        '\nIf you do not know a value, you can say "unknown".'
    )

    return "\n".join(lines)


# =========================================================
# FORMAT RETRIEVED EVIDENCE
# =========================================================

def format_evidence(docs):

    formatted = []

    for i, doc in enumerate(docs, 1):

        source = doc.metadata.get(
            "source_file",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        formatted.append(
            f"""
EVIDENCE {i}
File: {source}
Page: {page}

{doc.page_content}
"""
        )

    return "\n".join(formatted)


# =========================================================
# FILTER REFERENCE PAGES
# =========================================================

def is_reference_chunk(doc):

    text = doc.page_content.lower()

    et_al_count = text.count(
        "et al."
    )

    year_count = len(
        re.findall(
            r"\b(19|20)\d{2}\b",
            text
        )
    )

    if "references" in text[:200]:
        return True

    if (
        et_al_count >= 3
        and year_count >= 4
    ):
        return True

    return False


def retrieve_clean_evidence(
    query,
    k=10,
    final_k=4
):

    docs = vectorstore.similarity_search(
        query,
        k=k
    )

    clean_docs = [
        doc
        for doc in docs
        if not is_reference_chunk(doc)
    ]

    return clean_docs[:final_k]


# =========================================================
# MULTI-METRIC ANALYSIS
# =========================================================

def analyze_environmental_profile(profile):

    profile_text = json.dumps(
        profile.model_dump(
            exclude_none=True
        ),
        indent=2
    )

    docs = retrieve_clean_evidence(
        f"""
Environmental assessment for:

{profile_text}

Relationships between biodiversity,
soil organic carbon,
rainfall,
land use,
soil health,
and climate.
""",
        k=10,
        final_k=4
    )

    evidence = format_evidence(
        docs
    )

    prompt = f"""
You are a conservative environmental scientist.

USER ENVIRONMENTAL PROFILE:

{profile_text}

IMPORTANT:
Values in the profile come directly from the user.

Preserve their units EXACTLY.

For example:
0.3% SOC must remain 0.3%.
Do NOT convert 0.3% into g/kg.

Do NOT call a value "low", "very low", "high",
or abnormal unless:
1. the user described it that way, OR
2. supplied evidence provides a threshold.


RETRIEVED SCIENTIFIC EVIDENCE:

{evidence}


Analyze the interaction of MULTIPLE variables.

Use ONLY the supplied evidence for scientific factual claims.

Do not invent mechanisms.

Do not invent numerical effects.

Do not make local-to-global climate causal claims.

If a relationship is reasonable but not directly established,
begin the statement with:

"Inference:"


Use this structure:

### Environmental Diagnosis

Briefly summarize the environmental situation.

### Multi-Metric Interactions

Provide 3-4 important interactions.

For every interaction clearly distinguish:
- Evidence
- Inference

### Biodiversity Impact

Describe likely biodiversity implications,
remaining conservative.

### Evidence Used

List the exact PDF filename and page
for the important evidence used.
"""

    response = invoke_with_retry(
        llm,
        prompt
    )

    return response.content, docs


# =========================================================
# RECOMMENDATION RETRIEVAL
# =========================================================

def build_recommendation_evidence(profile):

    agro_docs = retrieve_clean_evidence(
        """
        agroforestry biodiversity conservation
        climate adaptation habitat connectivity water regulation
        """,
        k=12,
        final_k=3
    )

    management_docs = retrieve_clean_evidence(
        """
        Figure 13 conservation reduced tillage
        crop rotations cover cropping
        soil carbon sequestration management strategies
        """,
        k=12,
        final_k=4
    )

    all_docs = agro_docs + management_docs

    unique_docs = []
    seen = set()

    for doc in all_docs:

        key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            doc.page_content[:100]
        )

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs
# =========================================================
# RECOMMENDATION GENERATOR
# =========================================================
def generate_submission_recommendations(profile, diagnosis):

    profile_text = json.dumps(
        profile.model_dump(exclude_none=True),
        indent=2
    )

    docs = build_recommendation_evidence(profile)
    evidence = format_evidence(docs)

    prompt = f"""
You are an environmental decision-support assistant.

ENVIRONMENTAL PROFILE:
{profile_text}

RETRIEVED SCIENTIFIC EVIDENCE:
{evidence}

Generate exactly THREE recommendations:

1. Agroforestry
2. Cover Cropping / Crop Rotation
3. Conservation / Reduced Tillage

STRICT RULES:

- Use only the supplied evidence for factual scientific claims.
- Keep the response concise.
- Do not invent numbers, species, field sizes, or quantitative effects.
- Do not claim an intervention increases SOC unless the evidence directly supports it.
- Site-specific reasoning that goes beyond direct evidence must begin with "Inference:".
- Preserve user-provided units exactly.
- Every recommendation MUST contain all required sections.

Use EXACTLY this structure:

### Recommendation 1 – Agroforestry

**Direct Evidence:**
Brief evidence-supported statement.

**Relevance to this Site:**
Inference: brief site-specific reasoning.

**Targeted Metrics:**
- metric

**Time Horizon:**
Planning estimate only; not established by retrieved evidence.

**Confidence:**
High / Medium / Low confidence in retrieved evidence.

**Evidence:**
- filename, page


### Recommendation 2 – Cover Cropping / Crop Rotation

Use the same sections.


### Recommendation 3 – Conservation / Reduced Tillage

Use the same sections.


### Priority Order

Decision-support judgement, not a scientifically proven universal ranking.

1. ...
2. ...
3. ...

IMPORTANT:
Finish ALL THREE recommendations and the Priority Order.
Do not stop part-way through the response.
"""

    # Try up to 3 times if Groq produces an incomplete answer
    for attempt in range(3):

        response = invoke_with_retry(
            llm,
            prompt
        )

        content = response.content.strip()

        required_sections = [
            "Recommendation 1",
            "Recommendation 2",
            "Recommendation 3",
            "Priority Order"
        ]

        complete = all(
            section.lower() in content.lower()
            for section in required_sections
        )

        # Also reject suspiciously tiny outputs
        if complete and len(content) > 700:
            return content, docs

        print(
            f"Incomplete recommendation response "
            f"(attempt {attempt + 1}/3). Retrying..."
        )

    raise RuntimeError(
        "The recommendation model returned an incomplete response "
        "after 3 attempts. Please retry the assessment."
    )
# =========================================================
# EVIDENCE VERIFIER
# =========================================================

def check_groundedness(
    answer,
    docs
):

    evidence = format_evidence(
        docs
    )

    prompt = f"""
You are a strict scientific evidence verifier.

RETRIEVED EVIDENCE:

{evidence}


ANSWER:

{answer}


Evaluate ONLY scientific factual claims.

Rules:

- Paraphrasing is allowed.
- Tables, headings and figures count as evidence.
- Clearly labelled "Inference:" is allowed.
- Planning estimates are not scientific claims.
- Decision-support rankings are not scientific claims.
- Numerical claims need direct support.
- Specific mechanisms need direct support.
- Do not flag a claim merely because wording differs.

Return:

supported_claims:
important supported claims only.

unsupported_claims:
important unsupported factual claims only.

explanation:
one short explanation.
"""

    result = invoke_with_retry(
        grounding_llm,
        prompt
    )

    return result


# =========================================================
# NORMALIZE GROUNDING STATUS
# =========================================================
def determine_grounding_status(grounding):

    unsupported = grounding.unsupported_claims or []
    supported = grounding.supported_claims or []

    # No claims were identified at all:
    # this must NEVER be called Grounded.
    if len(supported) == 0 and len(unsupported) == 0:
        return "Verification Failed"

    if len(unsupported) == 0:
        return "Grounded"

    if len(supported) > 0:
        return "Partially Grounded"

    return "Not Grounded"
# =========================================================
# CACHED COMPLETE ASSESSMENT
# =========================================================

@lru_cache(maxsize=32)
def run_complete_assessment(profile_json):

    profile_data = json.loads(profile_json)

    profile = EnvironmentalProfile(**profile_data)

    diagnosis, diagnosis_docs = (
        analyze_environmental_profile(profile)
    )

    recommendations, rec_docs = (
        generate_submission_recommendations(
            profile,
            diagnosis
        )
    )

    # Verify ONLY the recommendation layer
    grounding = check_groundedness(
        recommendations,
        rec_docs
    )

    status = determine_grounding_status(
        grounding
    )

    response = f"""
## Environmental Assessment

{diagnosis}

---

## Evidence-Backed Recommendations

{recommendations}
"""

    return {
        "diagnosis": diagnosis,
        "recommendations": recommendations,
        "response": response,
        "grounding_status": status,
        "grounding_notes": grounding.unsupported_claims,
        "grounding_explanation": grounding.explanation
    }


# =========================================================
# COMPLETE CHATBOT PIPELINE
# =========================================================

def process_message(
    user_message,
    current_profile=None
):

    if current_profile:

        profile = EnvironmentalProfile(
            **current_profile
        )

    else:

        profile = EnvironmentalProfile()

    # Extract new information locally
    new_profile = (
        extract_environmental_profile(
            user_message
        )
    )

    # Merge with conversation memory
    profile = merge_profiles(
        profile,
        new_profile
    )

    missing = find_missing_fields(
        profile
    )

    # -----------------------------------------------------
    # ASK FOR MISSING INFORMATION
    # -----------------------------------------------------

    if missing:

        response = (
            generate_clarifying_question(
                profile
            )
        )

        return {
            "status": "needs_more_information",

            "profile":
                profile.model_dump(
                    exclude_none=True
                ),

            "response": response,

            "diagnosis": None,

            "recommendations": None,

            "grounding_status": None,

            "grounding_notes": [],

            "grounding_explanation": None
        }

    # -----------------------------------------------------
    # COMPLETE ASSESSMENT
    # -----------------------------------------------------

    profile_data = (
        profile.model_dump(
            exclude_none=True
        )
    )

    # Sorted JSON produces a stable cache key.
    profile_json = json.dumps(
        profile_data,
        sort_keys=True
    )

    result = run_complete_assessment(
        profile_json
    )

    return {
        "status": "complete",

        "profile": profile_data,

        "diagnosis":
            result["diagnosis"],

        "recommendations":
            result["recommendations"],

        "response":
            result["response"],

        "grounding_status":
            result["grounding_status"],

        "grounding_notes":
            result["grounding_notes"],

        "grounding_explanation":
            result["grounding_explanation"]
    }