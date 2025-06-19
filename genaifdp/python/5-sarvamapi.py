from sarvamai import SarvamAI
client = SarvamAI(
    api_subscription_key="sk_dd8qqhjm_YWBV6gIFh39XMBe66Q9wv0V4",
)
response = client.text.translate(
    input="Hi, My Name is Vinayak.",
    source_language_code="auto",
    target_language_code="te-IN",
    speaker_gender="Male"
)
response1 = client.text.translate(
    input="Hi, My Name is Vinayak.",
    source_language_code="auto",
    target_language_code="kn-IN",
    speaker_gender="Male"
)
response2 = client.text.translate(
    input="Hi, My Name is Vinayak.",
    source_language_code="auto",
    target_language_code="hi-IN",
    speaker_gender="Male"
)
print(response1)
print(response2)
print(response)
# response = client.text.summarize(
#    input="Hi, My Name is Vinayak.",
#    language_code="auto"
# )     