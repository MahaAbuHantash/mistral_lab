import os
import streamlit as st
from mistralai import Mistral, UserMessage

# ----------------------------
# 1) Mistral helper (same style as your lab)
# ----------------------------
def mistral(user_message, model="mistral-small-latest", is_json=False):
    # Lab code forces large model
    model = "mistral-large-latest"
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "ERROR: MISTRAL_API_KEY is not set."

    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=user_message)]
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content


# ----------------------------
# 2) Classification prompt (from lab)
# ----------------------------
CLASSIFIER_PROMPT = """
You are a bank customer service bot.
Your task is to assess customer intent and categorize customer
inquiry after <<<>>> into one of the following predefined categories:
card arrival
change pin
exchange rate
country support
cancel transfer
charge dispute
If the text doesn't fit into any of the above categories,
classify it as:
customer service
You will only respond with the predefined category.
Do not provide explanations or notes.
###
Here are some examples:
Inquiry: How do I know if I will get my card, or if it is lost? I am concerned about the delivery process and would like to ensure that I will receive my card
Category: card arrival
Inquiry: I am planning an international trip to Paris and would like to inquire about the current exchange rates for Euros as well as any associated fees for
Category: exchange rate
Inquiry: What countries are getting support? I will be traveling and living abroad for an extended period of time, specifically in France and Germany, and w
Category: country support
Inquiry: Can I get help starting my computer? I am having difficulty starting my computer, and would appreciate your expertise in helping me troubleshoot th
Category: customer service
###
<<<
Inquiry: {inquiry}
>>>
Category:
"""

def classify_intent(user_text: str) -> str:
    prompt = CLASSIFIER_PROMPT.format(inquiry=user_text)
    category = mistral(prompt).strip().lower()
    # Keep only expected labels (simple safety)
    allowed = {
        "card arrival", "change pin", "exchange rate", "country support",
        "cancel transfer", "charge dispute", "customer service"
    }
    return category if category in allowed else "customer service"


# ----------------------------
# 3) Response generator (simple + customizable)
# ----------------------------
def generate_reply(user_text: str, category: str) -> str:
    # You can expand these “facts/policies” creatively later
    bank_facts = """
- Card arrival: Cards usually arrive within the delivery window stated in the app. If delayed, we can help check status.
- Change PIN: You can change your PIN from the app/ATM depending on your card type.
- Exchange rate: Exchange rates vary by currency and time; fees may apply depending on the transfer type.
- Country support: Card usage/support varies by country; tell us which country you’re traveling to.
- Cancel transfer: Some transfers can be canceled only if they are still pending.
- Charge dispute: If you see an unrecognized charge, we can open a dispute after we verify the transaction details.
- Customer service: For anything else, we’ll guide you to the right support step.
"""

    prompt = f"""
You are a helpful customer support assistant.
The user’s message was classified as: {category}

Use the following support facts/policy:
{bank_facts}

Now answer the user clearly and concisely, and ask ONE follow-up question if needed.

User message: {user_text}
"""

    return mistral(prompt)


# ----------------------------
# 4) Optional summarizer (based on lab summarization idea)
# ----------------------------
def summarize_thread(messages) -> str:
    convo = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    prompt = f"""
Summarize the conversation below in 5-7 bullet points.
Conversation:
{convo}
"""
    return mistral(prompt)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Customer Support Chatbot (Mistral)", page_icon="💬")
st.title("💬 Customer Support Chatbot (Mistral)")

# Show whether key exists
if not os.getenv("MISTRAL_API_KEY"):
    st.warning("MISTRAL_API_KEY is not set. Set it in your environment before running the app.")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Type your question...")

if user_input:
    # store user msg
    st.session_state.chat.append({"role": "user", "content": user_input})

    # classify + reply
    category = classify_intent(user_input)
    assistant_reply = generate_reply(user_input, category)

    # store assistant msg (include category)
    st.session_state.chat.append({"role": "assistant", "content": f"**Category:** {category}\n\n{assistant_reply}"})

# render chat
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# summarization button
if st.session_state.chat:
    if st.button("Summarize conversation"):
        summary = summarize_thread(st.session_state.chat)
        st.info(summary)
