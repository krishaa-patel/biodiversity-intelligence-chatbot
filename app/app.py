import streamlit as st
from backend import process_message


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Biodiversity Intelligence",
    page_icon="🌿",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    margin-bottom: 1.5rem;
}

.hero-box {
    padding: 1.1rem 1.3rem;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    background: #f8faf9;
    margin-bottom: 1.4rem;
}

.example-box {
    background: #e7f1fb;
    border-radius: 8px;
    padding: 16px;
    height: 118px;
    box-sizing: border-box;
    color: #004b87;
}

.example-title {
    font-weight: 700;
    margin-bottom: 16px;
}

.example-text {
    line-height: 1.45;
    font-size: 0.95rem;
}

.complete-box {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.section-label {
    color: #6b7280;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "profile" not in st.session_state:
    st.session_state.profile = {}

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🌿 Biodiversity Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered environmental decision support using
    scientific retrieval, multi-metric reasoning,
    and evidence verification.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-box">
        <b>How it works:</b>
        Describe your land, farm, or ecosystem.
        The system collects environmental metrics,
        retrieves scientific evidence, reasons across
        multiple variables, and generates verified
        recommendations.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🌱 Environmental Profile")

    if st.session_state.profile:

        for key, value in st.session_state.profile.items():

            label = (
                key
                .replace("_", " ")
                .title()
            )

            with st.container(border=True):

                st.caption(label.upper())

                st.markdown(
                    f"**{value}**"
                )

    else:

        st.info(
            "Your environmental profile will "
            "appear here as information is collected."
        )

    st.divider()

    st.subheader("🔎 Evidence Verification")

    if st.session_state.latest_result:

        status = (
            st.session_state
            .latest_result["grounding_status"]
        )

        if status == "Grounded":

            st.success(
                "✅ Grounded"
            )

        elif status == "Partially Grounded":

            st.warning(
                "⚠️ Partially Grounded"
            )

        elif status == "Verification Failed":

            st.error(
                "⚠️ Verification Failed"
            )

        else:

            st.error(
                "❌ Not Grounded"
            )

        notes = (
            st.session_state
            .latest_result
            .get(
                "grounding_notes",
                []
            )
        )

        if notes:

            with st.expander(
                "Verification Notes"
            ):

                for note in notes:

                    st.write(
                        "•",
                        note
                    )

    else:

        st.caption(
            "Verification appears after "
            "a complete assessment."
        )

    st.divider()

    if st.button(
        "🔄 Reset Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.profile = {}
        st.session_state.latest_result = None

        st.rerun()


# =========================================================
# LANDING EXAMPLES
# =========================================================

if not st.session_state.messages:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            '<div class="example-box">'
            '<div class="example-title">'
            'Example 1'
            '</div>'
            '<div class="example-text">'
            'Biodiversity is declining on my wheat farm.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="example-box">'
            '<div class="example-title">'
            'Example 2'
            '</div>'
            '<div class="example-text">'
            'My soil organic carbon is 0.4% '
            'and rainfall is low.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            '<div class="example-box">'
            '<div class="example-title">'
            'Example 3'
            '</div>'
            '<div class="example-text">'
            'I manage agricultural land '
            'in a semi-arid region.'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# CONVERSATION HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# FINAL RESULT
# =========================================================

if st.session_state.latest_result:

    result = st.session_state.latest_result

    status = result["grounding_status"]

    st.markdown(
        f"""
        <div class="complete-box">
            <b>✅ Environmental Assessment Complete</b><br>
            Evidence status: <b>{status}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # ASSESSMENT
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-label">'
        'Environmental reasoning'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🧭 Environmental Assessment"
    )

    with st.container(border=True):

        st.markdown(
            result["diagnosis"]
        )

    st.write("")

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-label">'
        'Evidence-backed interventions'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🌱 Recommendations"
    )

    with st.container(border=True):

        st.markdown(
            result["recommendations"]
        )

    st.write("")

    # -----------------------------------------------------
    # VERIFICATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-label">'
        'Scientific grounding'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🔎 Evidence Verification"
    )

    if status == "Grounded":

        st.success(
            "✅ Grounded — no significant unsupported "
            "scientific claims were flagged."
        )

    elif status == "Partially Grounded":

        st.warning(
            "⚠️ Partially Grounded — the main analysis "
            "is supported, but some secondary claims "
            "require stronger direct evidence."
        )

    elif status == "Verification Failed":

        st.error(
            "⚠️ Verification could not evaluate enough "
            "scientific claims."
        )

    else:

        st.error(
            "❌ Not Grounded — important scientific "
            "claims require stronger evidence."
        )

    notes = result.get(
        "grounding_notes",
        []
    )

    if notes:

        with st.expander(
            "View verification notes"
        ):

            for note in notes:

                st.write(
                    "•",
                    note
                )

    explanation = result.get(
        "grounding_explanation"
    )

    if explanation:

        st.caption(
            explanation
        )


# =========================================================
# USER INPUT
# =========================================================

user_message = st.chat_input(
    "Describe your ecosystem, land conditions, "
    "or biodiversity concern..."
)


if user_message:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    # -----------------------------------------------------
    # RUN BACKEND
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Analyzing environmental information..."
        ):

            result = process_message(
                user_message,
                st.session_state.profile
            )

        # Update profile
        st.session_state.profile = (
            result["profile"]
        )

        # -------------------------------------------------
        # NEED MORE INFORMATION
        # -------------------------------------------------

        if (
            result["status"]
            == "needs_more_information"
        ):

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"]
            })

            # Any older result is no longer current
            st.session_state.latest_result = None

        # -------------------------------------------------
        # COMPLETE
        # -------------------------------------------------

        else:

            st.session_state.latest_result = {
                "diagnosis":
                    result["diagnosis"],

                "recommendations":
                    result["recommendations"],

                "grounding_status":
                    result["grounding_status"],

                "grounding_notes":
                    result["grounding_notes"],

                "grounding_explanation":
                    result[
                        "grounding_explanation"
                    ]
            }

            st.session_state.messages.append({
                "role": "assistant",
                "content":
                    "The environmental assessment is complete. "
                    "The structured results are shown below."
            })

        st.rerun()

    except Exception as e:

        st.session_state.messages.append({
            "role": "assistant",
            "content":
                "An error occurred while processing "
                f"the request:\n\n`{e}`"
        })

        st.rerun()