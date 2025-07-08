import os
import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm
import numpy as np
import time
import random

# Azure OpenAI Configuration
endpoint = "https://bvit-mahbh6h4-eastus2.cognitiveservices.azure.com/"
deployment = "gpt-4o-mini-benja"
model_name = "gpt-4o-mini"
subscription_key = "2hV4QJrXC1V4lMuJyBwcX5sV9wqva8UvgH8UM5KyNRJx9XtJuGEYJQQJ99BEACHYHv6XJ3w3AAAAACOGDncZ"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Load dataset and shuffle
df = pd.read_csv("dataset_recortado.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Base prompt defined previously (omitted here for brevity, keep the one you already built)
prompt_base = """
You are a highly rigorous crypto analyst and dataset curator.
 Your task is to decide whether a news article is suitable for training a model that learns the short-term impact of news on Bitcoin's price.

🔻 Inputs:
A news article


A target: the observed multiplicative price change (new_price ÷ old_price), measured from 1 hour before to 2 hours after publication



🧠 3-Step Analysis
🔹 Step 1 — Relevance
Respond RELEVANT: 1 only if the news:
✅ Describes a concrete, time-aligned event or decision
 ✅ Comes from an influential actor (e.g., govs, central banks, institutions, Musk, G7, major exchanges)
 ✅ Has clear potential to move price within 1–2 hours
 ✅ Is likely to trigger a position from traders or algos
Reject with RELEVANT: 0 if the news:
❌ Is backward-looking (e.g. price recap, trend commentary)
 ❌ Is gradual/expected (e.g. mining difficulty adjustment, slow adoption talk)
 ❌ Lacks clear intent or decision (e.g. sentiment, speculation)
 ❌ Would not trigger an actionable reaction

🔹 Step 2 — Directional Match (Strict Numerical Check)

If RELEVANT = 1:

Ask: Would a rational trader interpret this news as bullish (price should rise) or bearish (price should fall)?
Based on the DIRECTIONAL_REASONING, assign an EXPECTED_DIRECTION:
bullish → Bitcoin price is expected to rise
bearish → Bitcoin price is expected to fall
Then compare the EXPECTED_DIRECTION to the observed target (i.e., price change ratio):
✅ If EXPECTED_DIRECTION = bullish, then target must be strictly greater than 1.000000
✅ If EXPECTED_DIRECTION = bearish, then target must be strictly less than 1.000000
❌ If this condition is not met →

Set DIRECTIONAL MATCH: no
You must return FINAL DECISION: 0
⚠️ Do not rationalize mismatches. This is a strict check between expectation and observed market response.

Example: If the news is bullish but target = 0.984, the direction is invalid → DIRECTIONAL MATCH: no.



🔹 Step 3 — Algo Trader Trigger Test
Ask:
"Would this headline realistically cause an algorithmic or human day trader to take a position within minutes?"
If not → ALGO TRADER TRIGGER: no

✅ Final Label

Only return FINAL DECISION: 1 if:
- RELEVANT = 1
- DIRECTIONAL MATCH = yes
- ALGO TRADER TRIGGER = yes

Otherwise, return:
FINAL DECISION: 0


📦 Output Format (YAML)


REASONING: [Why the news is or isn't relevant — time-aligned, impactful, from credible actor?]
RELEVANT: [1 or 0]

DIRECTIONAL REASONING: [Short explanation on expected direction- why, bullish or bear]  
EXPECTED_DIRECTION: [bullish / bearish / N/A]  
DIRECTIONAL MATCH: [yes / no / N/A]

ALGO TRADER TRIGGER: [yes / no]  
FINAL DECISION: [1 or 0]


✅ Few-shot Examples (Strict Edition)
🟢 Example 1 — Clear regulatory move, aligned price drop
News: "U.S. Treasury freezes crypto exchange wallets in connection to sanctions evasion."  
Target: 0.985
REASONING: Concrete, time-aligned enforcement from a top-tier regulatory body. Directly impacts exchange operability and signals heightened regulatory risk.
RELEVANT: 1
DIRECTIONAL REASONING: Traders would interpret this as bearish — suggests increased enforcement risk and potential for exchange disruption or user fund freezing.
EXPECTED_DIRECTION: bearish
DIRECTIONAL MATCH: yes
ALGO TRADER TRIGGER: yes
FINAL DECISION: 1



🔴 Example 2 — Slow-moving technical update

News: "Bitcoin mining difficulty hits record high after algorithm adjustment."  
Target: 0.995
REASONING: Expected protocol-level change with no immediate surprise. Represents network strength but unfolds too slowly to trigger near-term price action.
RELEVANT: 0
DIRECTIONAL REASONING: N/A
EXPECTED_DIRECTION: N/A
DIRECTIONAL MATCH: N/A
ALGO TRADER TRIGGER: no
FINAL DECISION: 0



🔴 Example 3 — Clarification, not a decision

News: "President Bukele says Bitcoin use is voluntary."  
Target: 1.002
REASONING: Merely clarifies existing policy, not a new decision or concrete action. Does not materially change investor or trader posture.
RELEVANT: 0
DIRECTIONAL REASONING: N/A
EXPECTED_DIRECTION: N/A
DIRECTIONAL MATCH: N/A
ALGO TRADER TRIGGER: no
FINAL DECISION: 0



🟠 Example 4 — Whale alert, no confirmed action

News: "Whale Alert: Dormant wallet with 300 BTC (~$29.8M) reactivated after 11.6 years."  
Target: 0.99665
REASONING: Wallet movement alone lacks clear intent or time-aligned market action. No exchange transfer or sale confirmed. Speculative at best.
RELEVANT: 0
DIRECTIONAL REASONING: N/A
EXPECTED_DIRECTION: N/A
DIRECTIONAL MATCH: N/A
ALGO TRADER TRIGGER: no
FINAL DECISION: 0


🔴 Example 5 — Backward-looking analyst data

News: "CryptoQuant: Bitcoin demand cooling after surge to $112K; ETF buying slows, short positions increase."  
Target: 1.0100
REASONING: Post-hoc analysis of market positioning. No new action or decision, not time-aligned. Informational, not catalytic.
RELEVANT: 0
DIRECTIONAL REASONING: N/A
EXPECTED_DIRECTION: N/A
DIRECTIONAL MATCH: N/A
ALGO TRADER TRIGGER: no
FINAL DECISION: 0



🟢 Example 6 — Major executive order

News: "Donald Trump signs executive order to create U.S. Bitcoin reserve using 200,000 seized BTC."  
Target: 1.0055
REASONING: Concrete executive action from a high-influence figure. Directly related to Bitcoin accumulation and state-level policy shift. Timed and impactful.
RELEVANT: 1
DIRECTIONAL REASONING: Bullish — implies demand absorption and a signal of state-level BTC endorsement, likely prompting buying behavior.
EXPECTED_DIRECTION: Bullish
DIRECTIONAL MATCH: yes
ALGO TRADER TRIGGER: yes
FINAL DECISION: 1



🔴 Example 7 — Relevance, but mismatch

News: "Jake Sullivan highlights crypto's role in ransomware ahead of G7/NATO summits."  
Target: 1.0001
REASONING: Statement from a key national security figure ahead of regulatory-relevant summits. Strong bearish implications due to criminal finance framing.
RELEVANT: 1
DIRECTIONAL REASONING: Bearish — signals potential crackdown or international coordination on crypto restrictions or surveillance.
EXPECTED_DIRECTION: bearish
DIRECTIONAL MATCH: no
ALGO TRADER TRIGGER: yes
FINAL DECISION: 0


🔴 Example 8 — Relevance, but mismatch
News:"A crackdown on crypto staking from the SEC has weighed on shares of Coinbase ( COIN ) this week following charges levied at competitors and a cryptic tweet from CEO Brian Armstrong. Shares of Coinbase fell 4.2% on Friday following a 13% plunge Thursday that has seen the stock forfeit roughly half of its year-to-date rally. Still, shares of Coinbase are up more than 60% in 2023. Concerns from investors come after competitor exchange Kraken paid $30 million to settle charges levied by the U.S. Securities and Exchange Commission the company offered unregistered securities through its staking program. As part of the settlement, Kraken agreed to shutter its staking program for U.S. customers while neither admitting nor denying the SEC's allegations. The SEC's settlement with Kraken came just hours after Coinbase CEO Brian Armstrong warned Wednesday night of "rumors that the SEC would like to get rid of crypto staking in the U.S. for retail customers." Ether and other cryptocurrencies that use staking, such as Cardano and Solana, were down by at least 6% in the last 24 hours. For the same period, the total market capitalization for crypto assets has shed 4.6% or $49 billion. Staking is an alternative"
Target: 1.0075891685653473
REASONING: The news describes a concrete event involving regulatory action from the SEC, which has significant implications for the crypto market, particularly for Coinbase and staking services. This is time-aligned and has the potential to impact Bitcoin's price within the specified timeframe.
RELEVANT: 1
DIRECTIONAL REASONING: The news is bearish as it indicates increased regulatory scrutiny and potential restrictions on staking, which could lead to a decline in investor confidence and price.
EXPECTED_DIRECTION: bearish  
DIRECTIONAL MATCH: NO

ALGO TRADER TRIGGER: N/A
FINAL DECISION: 0


""".strip()  # use the complete prompt_base you defined before

# Function to classify
def classify_news(news_text, price_change):
    full_prompt = prompt_base + f"\n\nNews: {news_text.strip()}\nTarget: {price_change}\nFINAL DECISION:"
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Respond exactly in the YAML format specified in the prompt."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=250,
            temperature=0
        )
        output = response.choices[0].message.content.strip()
        return output
    except Exception as e:
        print("❌ Error with news:", news_text[:60], "|", str(e))
        return None

# Storage
relevant_news = []
saved_block = 0
target = 10_000

# Iteration
for idx, row in tqdm(df.iterrows(), total=len(df)):
    if len(relevant_news) >= target:
        break

    text = row["article_text"]
    target_value = row["target"]
    
    reasoning = classify_news(text, target_value)
    if reasoning and "FINAL DECISION: 1" in reasoning:
        # Extract only the REASONING field
        has_reasoning = "REASONING:" in reasoning
        reasoning_text = ""
        if has_reasoning:
            try:
                reasoning_text = reasoning.split("REASONING:")[1].split("\n")[0].strip()
            except:
                reasoning_text = ""

        relevant_news.append({
            "article_text": text.strip(),
            "REASONING": reasoning_text,
            "target": target_value
        })

        # Save every 1000 news items
        if len(relevant_news) % 1000 == 0:
            df_to_save = pd.DataFrame(relevant_news)
            df_to_save.to_csv(f"filtered_news_block_{saved_block}.csv", index=False)
            print(f"\n💾 Partial save with {len(relevant_news)} news items in block {saved_block}")
            saved_block += 1

# Save the total if not exact in multiples of 1000
if len(relevant_news) % 1000 != 0:
    df_to_save = pd.DataFrame(relevant_news)
    df_to_save.to_csv(f"filtered_news_block_{saved_block}.csv", index=False)
    print(f"\n💾 Final save with {len(relevant_news)} news items in total")

print(f"\n✅ Process completed. Total accepted news: {len(relevant_news)}") 