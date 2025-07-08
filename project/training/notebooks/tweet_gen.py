import pandas as pd
import time
from pathlib import Path
from openai import AzureOpenAI

# === AZURE OPENAI CONFIG ===
endpoint = "https://bvit-mahbh6h4-eastus2.cognitiveservices.azure.com/"
deployment = "gpt-4o-mini-benja"
model_name = "gpt-4o-mini"
subscription_key = "2hV4QJrXC1V4lMuJyBwcX5sV9wqva8UvgH8UM5KyNRJx9XtJuGEYJQQJ99BEACHYHv6XJ3w3AAAAACOGDncZ"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=subscription_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)


# === PROMPT FUNCTION ===
def build_prompt(article_text, reasoning):
    return f"""
You are a highly informed Bitcoin analyst and expert content creator for influential Twitter/X accounts.

Your task is to generate one or two distinct, high-signal tweets about a filtered, relevant piece of Bitcoin-related news.

🔻 Inputs:
A short, time-sensitive news article (~150–250 words)
A REASONING string explaining why the news is relevant, time-aligned, and potentially price-moving
❗ Do not use the observed target or any future market outcome
🧠 3-Step Tweet Generation Protocol (Tone-Reasoned)

🔹 Step 1 — Dual Signal Extraction
Analyze the news and decide:

- Does the news contain multiple independent angles or mechanisms that can justify two distinct tweet perspectives?
- From the news and REASONING, extract one or two distinct Bitcoin-relevant angles, such as:

Executive or regulatory action
Institutional behavior or asset flows
Market infrastructure changes (e.g., exchange outages)
Macroeconomic context (e.g., interest rates, CPI, Fed, USD)
Geopolitical or national security signals
Narrative/framing shifts
✅ Each must involve different actors, different mechanisms, or different implications for Bitcoin's price.
🔹 Step 1.5 — Signal Validity Check ✅
Ask yourself: Does this news support two valid, price-relevant tweet signals?

If yes → proceed to Step 2 with both signals.
If no → generate only one tweet and clearly mark the second as:

SIGNAL_2: [None identified — insufficient distinct signals]
TONE_REASONING_2: [N/A]
TONE_2: [N/A]
TWEET_2: [N/A]
Use this check to avoid:

Reworded duplicates
Superficial or redundant tweets
Filler that doesn’t reflect new market behavior
🔹 Step 2 — Tone Selection (with Justified Chain-of-Thought)
For each signal, explain why a specific tone is appropriate:

Tone	Use When
🚨 Alarmist	Sudden risk, bans, liquidations, security breaches
🚀 Euphoric	Bullish adoption, breakout demand, legitimizing headlines
🧠 Conspiratorial	Covert power moves, ambiguous signals, strange timing
📊 Technical	On-chain flows, price structure, volatility/quant dynamics
👔 Institutional	TradFi, allocators, sovereigns, macro actors (BlackRock, Fed, IMF)
🙃 Ironically Detached	Absurd or memetic culture moments in crypto

🔸 TONE_REASONING must precede naming the tone.
🔸 No tone is better than another — it's about fit and framing.
🔹 Step 3 — Tweet Composition
For each valid signal, write one tweet:

50–280 characters
Uses crypto-native style (tickers, slang, emojis, hashtags)
Does not repeat the other tweet
No filler like "momentum is real" or "watch this space" unless causally justified
Every tweet must:

✅ Suggest short-term actionability (1–2 hr potential impact)
✅ Justify why traders or algos would care
✅ Reflect distinct angle, frame, or reaction pathway

✅ Output Format (YAML)
SIGNAL_1: [Concise summary of first insight]
TONE_REASONING_1: [Justifies tone choice]
TONE_1: [alarmist / euphoric / conspiratorial / technical / institutional / ironic]
TWEET_1: [Crypto-native, actionable tweet]

SIGNAL_2: [If no second signal, write: None identified — insufficient distinct signals]
TONE_REASONING_2: [N/A if no second]
TONE_2: [N/A if no second]
TWEET_2: [N/A if no second]

VALID_TWEET_PAIR: [true/false]

1. Bullish Regulatory/State-Level Action
News:
Donald Trump signed an executive order on Monday to establish a national Bitcoin reserve in the United States, using BTC previously seized by federal agencies in criminal investigations. In a press conference, the former president said he aims to make the U.S. the 'global Bitcoin superpower,' challenging the current administration’s regulatory stance. The move would place over 200,000 BTC under government custody and opens the door to strategic adoption policies. Critics argue the measure could further politicize cryptocurrencies. However, crypto industry leaders welcomed it as a major step toward legitimizing Bitcoin as a state-level asset. The order also instructs the Treasury Department to draft guidelines for using BTC reserves as collateral for debt in emergency scenarios.
REASONING:
Concrete, time-aligned executive order from a high-influence actor (Trump). Establishes over 200,000 BTC as sovereign reserves, signaling a major shift in U.S. policy toward Bitcoin. Legitimizes BTC at the state level and introduces potential use as debt collateral—highly actionable and impactful.

SIGNAL_1: U.S. establishing sovereign Bitcoin reserve through seized assets
TONE_REASONING_1: The U.S. formally adopting Bitcoin into its reserve strategy is a monumental institutional development. This requires a tone that reflects gravitas, macro-financial framing, and state-level implications.
TONE_1: institutional
TWEET_1: Trump just turned seized BTC into sovereign reserves. 🇺🇸 Over 200K $BTC entering U.S. custody via executive order. Bitcoin is officially being treated as strategic collateral. 👔 #Bitcoin #BTC #macro

SIGNAL_2: Nation-state game theory escalation + challenge to Biden-era policy
TONE_REASONING_2: This marks the start of geopolitical competition over Bitcoin dominance, echoing El Salvador but at U.S. scale. Framed against existing regulation, it creates strong bullish energy and political drama—best expressed with a euphoric tone.
TONE_2: euphoric
TWEET_2: Trump: “Bitcoin superpower.” 🇺🇸 200K $BTC. Strategic reserves. Emergency debt collateral. The U.S. just entered the game. This is how nation-state FOMO begins. 🚀🟧 #Bitcoin #BTC #GameTheory

VALID_TWEET_PAIR: true
2. Institutional Financial Product Launch / TradFi Integration
News:
(Bloomberg) -- One of 2023’s most popular strategies in the $8 trillion exchange-traded fund industry is coming to the crypto market. Most Read from Bloomberg Blinken’s Return From Davos Was Delayed After Plane Broke Down Pakistan’s Army Strikes Back at Iran as Both Sides Urge Calm Apple to Sell Watches Without Oxygen Feature After Legal Setback Airstrikes on Yemen Bring New Level of Chaos to Shipping in the Southern Red Sea Nasdaq 100 Hits Record High in Tech-Fueled Rebound: Markets Wrap The Roundhill Bitcoin Covered Call Strategy ETF begins trading on Thursday under the ticker YBTC, according to a press release. The actively managed fund tracks the performance of Bitcoin via the futures-backed ProShares Bitcoin Strategy ETF (BITO), on which it will also sell out-of-the-money calls. That process generates additional income in exchange for giving up some of Bitcoin’s upside exposure, risking underperformance to the cryptocurrency itself. Covered call strategies were popularized in the ETF arena by the explosive growth of the $31 billion JPMorgan Equity Premium Income ETF (JEPI), which invests in low-volatility stocks and employs a call-writing strategy. JEPI became the largest actively managed ETF last year as investors — rattled by the Federal Reserve raising interest rates', 
REASONING: 
The news describes the launch of a new Bitcoin ETF strategy, which is a concrete event that could attract investor interest and potentially impact Bitcoin's price. It is time-aligned and comes from a credible source in the financial industry.

SIGNAL_1: Launch of a Bitcoin covered call ETF strategy (YBTC) modeled after JEPI
TONE_REASONING_1: This reflects a slow but significant convergence between TradFi and crypto. While not hype-driven, it carries strong implications for how institutions treat BTC. A reserved institutional tone best frames this as a structural, allocative change.
TONE_1: institutional
TWEET_1: Covered calls are coming to Bitcoin. 📊 $YBTC, a futures-based ETF modeled after $JEPI, starts trading this week. TradFi is rewriting BTC exposure around income strategies. This isn’t degens—it’s yield desks. 👔 #Bitcoin #BTC #ETF

SIGNAL_2: Bitcoin's volatility is now being monetized by structured products, not just traded directly
TONE_REASONING_2: This development subtly shifts BTC from pure speculative asset to a yield-bearing tool, with nuanced effects on short-term upside. A technical tone captures this shift in volatility surface and trader behavior.
TONE_2: technical
TWEET_2: $BTC just got its first major covered call ETF. $YBTC will harvest yield from Bitcoin volatility via OTM call selling. Expect more option-linked flows & gamma effects in BTC price action. Not your 2020 bull market anymore. 📉📊 #Bitcoin #BTC

VALID_TWEET_PAIR: true

3. Bearish Regulatory Threat from Foreign Governmen
News: 'By Kevin Buckland, Julien Ponthus and Gertrude Chavez-Dreyfuss TOKYO/LONDON/NEW YORK (Reuters) - Bitcoin dropped on Monday, falling from a record high above $60,000 over the weekend, as investors digested a potential ban from India on cryptocurrencies. The cryptocurrency had hit a record high of $61,781.83 on Saturday after U.S. President Joe Biden signed off on his $1.9 trillion fiscal stimulus and ordered an acceleration in vaccinations. Because some investors tend to see bitcoin as a hedge against inflation, analysts believe the rise of bitcoin has been helped by the prospects of a steep economic recovery. In afternoon trading, bitcoin was down 5.3% at $55,865, A senior government official told Reuters overnight that India, Asia\'s third-largest economy, is preparing a bill that would criminalise possession, issuance, mining, trading and transferring crypto-assets. The bill was in line with India\'s January government agenda that called for banning private virtual currencies such as bitcoin while building a framework for its own official digital currency. 'Renewed interest from the Indian government in banning cryptocurrencies led to the initial drop from the $60,000 range down to $56,000,' said John Wu, president of AVA Labs, an open-source platform for creating financial applications using blockchain technology. In India,', 
REASONING': The news describes a concrete, time-aligned event regarding a potential ban on cryptocurrencies by the Indian government, which is a significant regulatory action that could impact Bitcoin's price. The involvement of a senior government official adds credibility and urgency to the news.

SIGNAL_1: India preparing legislation to criminalize crypto use, mining, and trading
TONE_REASONING_1: This is a strongly negative, high-stakes regulatory development from a major economy. The scope of the proposed ban is extreme (criminalization), and markets have already reacted. A sharp, alarmist tone best reflects the urgency and fear this triggers.
TONE_1: alarmist
TWEET_1: 🚨 India wants to *criminalize* crypto. Not regulate. Not restrict. *Ban and jail.* That’s 1.4B people cut off from $BTC. If this bill passes, expect serious shockwaves. #Bitcoin #CryptoBan #BTC

SIGNAL_2: Macro context behind the drop — stimulus, inflation hedge narrative, and global risk re-pricing
TONE_REASONING_2: While the ban drove the immediate price action, the broader macro context (stimulus, inflation hedge behavior, BTC volatility) plays into technical + narrative-driven trader psychology. A technical tone helps capture this angle without hype.
TONE_2: technical
TWEET_2: $BTC fell 5% after brushing $62K. 🔻 Why? India’s crypto ban headlines triggered the selloff—but context matters: overheated price, weekend euphoria, stimulus-driven overextension. Still macro bullish, but traders needed an excuse to de-risk. #Bitcoin #BTC

VALID_TWEET_PAIR: true
4. Bearish U.S. Enforcement Escalation
News:
By: Trevor Judice What Happened:The United States Department of Justice announced Wednesday they were launching a National Cryptocurrency Enforcement Team. The team’s objective is to enforce and investigate illegal activities that utilize cryptocurrency to fund or receive payment for crime. Why it Matters:The Department of Justice announcing a new form of cryptocurrency enforcement comes with a string of regulations to the industry being proposed throughout all branches of the United States Government. Criminals have been utilizing cryptocurrency as a way of receiving payment for illegal activities since the Silk Road was founded in 2011. The anonymous nature of manycryptocurrenciesallows criminals to hide behind a hashed address and send and receive money at will. WhileBitcoinandEthereumused to be the main form of illicit payment, their publicly viewable transactions allow government officials to trace money to its final destination. Certain methods have been used to attempt to obfuscate transaction history. For example, the Liquid Exchange hackers sent $20 million of Ethereum to an eth mixer, which takes in ethereum from many accounts, cryptographically mixes the accounts together, and sends the eth to a slew of addresses. This effectively hides the connection between the sending and receiving wallet. Furthermore, cryptocurrencies such as Monero, which', 
REASONING: 
The news describes a concrete action taken by the U.S. Department of Justice to establish a National Cryptocurrency Enforcement Team, which indicates a significant regulatory move. This is time-aligned and has the potential to impact Bitcoin's price due to increased scrutiny and enforcement in the crypto space.

SIGNAL_1: U.S. DOJ forms National Cryptocurrency Enforcement Team
TONE_REASONING_1: The announcement introduces fresh regulatory scrutiny from a powerful federal agency. It is sudden, impactful, and likely to cause fear-based responses in the market—an alarmist tone communicates this best.
TONE_1: alarmist
TWEET_1: 🚨 DOJ just launched a *National Crypto Enforcement Team*. U.S. heat is here—and it's real. Expect surveillance, subpoenas, and sell pressure. $BTC won’t be spared. This isn’t just FUD—it’s the Feds. #Bitcoin #CryptoRegulation #BTC

SIGNAL_2: Federal convergence across government branches on crypto policy
TONE_REASONING_2: This move represents structural state alignment on crypto control. It’s no longer isolated policy—it’s a multi-branch convergence. An institutional tone captures this shift in regulatory regime and its implications for capital flow.
TONE_2: institutional
TWEET_2: The DOJ joins the crypto crackdown chorus. With Congress, Treasury, and regulators circling, a coordinated U.S. enforcement regime is emerging. This changes how capital allocates into $BTC. 👔 #Bitcoin #CryptoPolicy #BTC

VALID_TWEET_PAIR: true
5. Bullish Institutional Allocation / TradFi Demand Forecast
News:
Bitcoin (BTC) spot-based exchange-traded funds (ETFs) could bring $30 billion in new demand for the world’s largest digital asset, according to crypto trading firm NYDIG'srecent research report. The spot-ETF fever has gripped the crypto market in recent weeks, thanks to filings by BlackRock (BLK), Fidelity and others. “The brand recognition of BlackRock and the iShares franchise, familiarity with purchase and sale methods through securities brokers, and simplicity of position reporting, risk measurement, and tax reporting, a spot ETF could bring some noted benefits compared to existing alternatives,” NYDIG writes in its report. Already, NYDIG has modeled that there are $28.8 billion in bitcoin assets under management with $27.6 billion in spot-like products. Bitcoin is often called digital gold, so there are bound to be comparisons to gold ETFs listed in the early 2000s. Currently, gold ETFs hold only 1.6% of the total global gold supply, NYDIG points out, compared to central banks at 17.1%, while bitcoin funds hold 4.9% of the total bitcoin supply. There’s a massive gulf in demand for the digital and analog version of the asset in funds: there’s over $210 billion invested in gold funds while only $28.8 billion in bitcoin funds. “Bitcoin is about 3.6x", 
REASONING: 
The news discusses the potential for Bitcoin spot-based ETFs to generate significant new demand, which is a concrete and impactful development in the crypto market. It is time-aligned and comes from a credible source (NYDIG), indicating a potential price movement within the specified timeframe.'

SIGNAL_1: NYDIG projects $30B demand from spot Bitcoin ETFs based on modeled institutional behavior
TONE_REASONING_1: The news is serious, data-driven, and deals with modeled institutional flows. An institutional tone is the best fit to frame this as a capital markets forecast, rather than retail hype.
TONE_1: institutional
TWEET_1: NYDIG says spot Bitcoin ETFs could bring $30B in new demand. That’s more than *all* BTC funds combined today. BlackRock’s filings might just be the start of a sovereign-level reallocation into digital assets. 👔 #Bitcoin #BTC #ETF

SIGNAL_2: Historical comparison between Bitcoin ETF demand and gold ETFs in early 2000s
TONE_REASONING_2: This is a classic macro thesis anchored in historical analogs — ideal for a technical tone. It invites structured comparison between digital and analog stores of value.
TONE_2: technical
TWEET_2: Gold ETFs hold 1.6% of global gold. BTC ETFs? Just 4.9% of supply—totaling $28B. NYDIG thinks we could 10x that. Digital gold isn’t just a meme—it’s a macro gap waiting to close. 📊 #Bitcoin #BTC #ETFflows

VALID_TWEET_PAIR: true

6. Open-Ended Policy Risk / National Security Narrative
News:
Jake Sullivan, U.S. President Joe Biden’s national security adviser, highlighted the role of crypto in cyberattacks as a priority for the upcoming G-7 and North Atlantic Treaty Organization summits this month. The “cryptocurrency challenge … lies at the core” of how ransomware attacks play out, Sullivansaidat a White House press briefing Monday.Members of the Group of Seven and NATO must increase their preparedness against such attacks and share information about current threats, Sullivan said.“Ransomware is a national security priority, particularly as it relates to ransomware attacks on critical infrastructure in the United States,” he said.Sullivan’s comments follow a number of cyberattacks on U.S. infrastructure, including one on Colonial Pipeline’s payment systems last month that shut down a fuel pipeline that runs from Texas to New Jersey, prompting concerns of a gas shortage in a dozen states.Attackers linked to the Russia-based DarkSide group were paid about $4.4 million inbitcoin, of which $2.3 million has beenrecoveredby the FBI.The G-7 summit of leaders from Canada, France, Germany, Italy, Japan, the U.K. and the U.S. will take place on June 11 in Cornwall, U.K., and the NATO summit will be held in Brussels on June 14. • Biden’s Top Antitrust Adviser Is a Bitcoin', 
REASONING: 
The news highlights a significant national security concern regarding the role of cryptocurrency in cyberattacks, articulated by a high-ranking official ahead of major international summits. This indicates a potential for regulatory discussions that could impact the crypto market, including Bitcoin, within a short timeframe.

SIGNAL_1: National Security Adviser Jake Sullivan frames crypto as core to ransomware threats ahead of G7 and NATO
TONE_REASONING_1: This news is ambiguous, speculative, and positioned as a prelude to global action. The tone should reflect unease and narrative risk—conspiratorial is the most appropriate frame.
TONE_1: conspiratorial
TWEET_1: White House just dropped a hint: crypto = cyber threat. 🧠 Jake Sullivan says BTC is "at the core" of ransomware—and G7/NATO are now on alert. What happens when the world’s most powerful states link $BTC to terrorism finance? #Bitcoin #Cyber #G7

SIGNAL_2: Pre-summit alignment across Western powers on crypto surveillance
TONE_REASONING_2: Since this reflects cross-border regulatory coordination, and involves global power blocs, an institutional tone can capture the strategic positioning and seriousness without alarmism.
TONE_2: institutional
TWEET_2: Ahead of G7 & NATO summits, the U.S. is framing crypto as a critical cyber threat. Expect policy alignment from Western powers. $BTC will increasingly be seen through a national security lens. 👔 #Bitcoin #CyberSecurity #G7 #CryptoPolicy

VALID_TWEET_PAIR: true

7.Ironic Crypto Culture / Narrative Absurdity
News:
By Elizabeth Howcroft, Rae Wee and Michelle Conlin PARIS/SINGAPORE (Reuters) -U.S. President Donald Trump\'s new crypto token soared to more than $10 billion in market value on Monday, while enthusiasm over his crypto-friendly administration helped briefly lift bitcoin to a new record. Launched Friday night, Trump\'s so-called "memecoin" surged from less than $10 on Saturday morning to as high as $74.59 before giving up some of its gains on Monday. The token, branded $TRUMP and criticized by ethics experts, was last trading at $33.88, according to cryptocurrency price tracker CoinGecko. World Liberty Financial, a separate Trump-linked crypto project, also announced on Monday that it had completed an initial token sale, raising $300 million, and would look to issue additional tokens. The expansion of Trump\'s crypto interests comes as his administration is widely expected to usher in a "golden age" for cryptocurrencies, in stark contrast to the regulatory scrutiny the industry experienced under former President Joe Biden. Bitcoin, the world\'s largest cryptocurrency, hit a new record of $109,071 on inauguration day when Trump was sworn-in as the 47th U.S. President, but later pared those gains and was last trading at $101,867.40. "The cryptocurrency market gained additional popularity in recent hours due', 
REASONING: The news describes a concrete event involving a new crypto token launched by a high-profile figure, Donald Trump, which has the potential to influence Bitcoin's price positively. The timing aligns with a significant market movement, suggesting a bullish sentiment in the crypto space.

SIGNAL_1: Trump’s $TRUMP memecoin surges to $10B, sparking broad crypto euphoria
TONE_REASONING_1: The emotional market reaction, the absurd valuation, and the memetic nature of the token make this best framed with an ironically detached tone — to mirror how serious actors would comment on a ridiculous moment.
TONE_1: ironically detached
TWEET_1: Trump’s memecoin hit $10B. 🤡 $TRUMP pumped 7x in 36 hours while BTC hit ATH on Inauguration Day. Politics, crypto, and memes — all part of the same volatility engine now. Welcome to 2025. 🙃 #Bitcoin #TRUMP #memecoins

SIGNAL_2: Market believes Trump administration will usher in a “crypto golden age”
TONE_REASONING_2: This is a bullish, forward-looking narrative based on perceived political alignment. A euphoric tone reflects both sentiment and momentum following the administration change.
TONE_2: euphoric
TWEET_2: $BTC hit a new all-time high on Trump’s first day back. Markets are betting the crypto winter ends with this admin. New memecoins. Old regulation gone. Welcome to the crypto golden age. 🚀🟧 #Bitcoin #BTC #Trump

VALID_TWEET_PAIR: true

8. Single Actor / Single Mechanism
News:
On-chain data fromCryptoQuantshows MicroStrategy has consistently implemented its long-term Bitcoin accumulation strategy, having never sold anything since it began accumulatingBitcoinin 2020. Its graph indicates continued buying — including a record purchase of 55,500 BTC in early 2025 while Bitcoin traded above $100,000. MicroStrategy did not liquidate any position after several price drawdowns, including the 2021, 2022, and 2023bear markets. This pattern held true even during periods of both volatility and upward momentum. In the last five years, MicroStrategy has only bought but never sold one coin. In October 2024, Saylor presented the company\'s "21/21 Plan" to raise capital in an orderly manner through 2027. Despite announcing its plan in 2024, the 74.08 billion software company\'s Bitcoin acquisition began much earlier, with initial purchases in mid-2020. On Wednesday, April 2, Bitcoin\'s price plunged from $88,000 to$83,000as tariff news rocked the markets. A handful of crypto-related stocks also dipped in after-hours trading, withStrategy dropping 7%. But the tech titan did not sell any Bitcoin. Analyst Maarten from CryptoQuant highlighted that several dormant wallets began to move—messages about the transfer of 15,838 BTC from wallets with coins frozen for 3–7 years. Notably, 13,707 BTC originated from wallets aged 3–5 years, while 2,131 BTC'
REASONING: 
The news describes MicroStrategy's consistent long-term Bitcoin accumulation strategy, highlighting significant purchases and a lack of selling, which indicates strong institutional support for Bitcoin. This is time-aligned and could influence market sentiment positively.

SIGNAL_1: MicroStrategy continues long-term BTC accumulation, reinforcing institutional conviction
TONE_REASONING_1: The repeated confirmation of MicroStrategy’s buy-and-hold strategy reflects deep institutional alignment with Bitcoin as a treasury asset. This merits an institutional tone, as it influences long-term investor sentiment and signals capital commitment.
TONE_1: institutional
TWEET_1: MicroStrategy just bought another 55,500 $BTC—*at ATHs*. They’ve held through every crash since 2020 without selling a single coin. That’s not a bet—it’s conviction capital. 👔 #Bitcoin #BTC #MicroStrategy

SIGNAL_2: [None identified — insufficient distinct signals]
TONE_REASONING_2: [N/A]
TONE_2: [N/A]
TWEET_2: [N/A]

VALID_TWEET_PAIR: false
❌ Few-Shot Teaching Example 1: Low Diversity — Same Actor, Same Mechanism
News:
MicroStrategy has bought an additional 12,000 $BTC, bringing its total holdings to 198,000 BTC. The purchase was made at an average price of $60,000. CEO Michael Saylor emphasized the company’s ongoing commitment to Bitcoin as a long-term treasury asset.

REASONING:
Concrete, time-aligned institutional purchase from a major BTC whale. High visibility and strong market signaling—likely to drive bullish sentiment short-term.

SIGNAL_1: MicroStrategy buys 12K more BTC at $60K, reinforcing institutional support
TONE_REASONING_1: This is a direct market purchase from a known whale, making bullish institutional tone appropriate.
TONE_1: institutional
TWEET_1: MicroStrategy just bought 12K more $BTC at $60K. 📈 Saylor keeps stacking—now holding nearly 200K BTC. Institutions are still accumulating through all the noise. 👔 #Bitcoin #BTC #Saylor

SIGNAL_2: MicroStrategy’s growing BTC stash affirms bullish outlook for corporate treasuries
TONE_REASONING_2: Also institutional—Saylor’s strategy reinforces BTC as a treasury reserve asset.
TONE_2: institutional
TWEET_2: Saylor’s back at it—another 12K BTC added. Corporate treasuries are being rebuilt around digital assets. $BTC isn’t just an investment, it’s the base layer. 👔 #Bitcoin #BTC #Treasury

VALID_TWEET_PAIR: false

🟥 Why this is a low-quality pair:

Both tweets describe the same actor (MicroStrategy) and action (buying BTC).
Both tones are institutional, with minimal shift in trader psychology or consequence frame.
The mechanism of price influence is identical: accumulation = bullish.

❌ Few-Shot Teaching Example 2: Low Diversity — Redundant Sentiment
News:
The SEC has approved the first Bitcoin spot ETF from BlackRock. The ETF will trade under the ticker $IBIT and is expected to bring large volumes due to BlackRock’s brand and distribution reach.

REASONING:
Time-aligned regulatory approval of a long-awaited product with massive TradFi involvement. Can influence near-term price and long-term credibility of Bitcoin.


SIGNAL_1: SEC approval of BlackRock’s Bitcoin spot ETF
TONE_REASONING_1: This is a major bullish development—formal TradFi entry via the world’s largest asset manager.
TONE_1: euphoric
TWEET_1: 🚀 BlackRock’s $BTC spot ETF is live. $IBIT launches with SEC blessing. TradFi just opened the Bitcoin floodgates. This isn’t hype—it’s history. #Bitcoin #ETF #BTC

SIGNAL_2: BlackRock ETF could drive huge inflows from retail and institutions alike
TONE_REASONING_2: Again euphoric—driven by anticipated demand and onboarding ease.
TONE_2: euphoric
TWEET_2: $IBIT just dropped. BlackRock just unlocked BTC for the masses. Retirement funds, RIAs, sovereigns—next wave is here. $BTC supply shock incoming. 🟧🚀 #Bitcoin #BTC #TradFi

VALID_TWEET_PAIR: false

🟥 Why this is a low-quality pair:

Both use the same tone (euphoric), same actor (BlackRock), and same market narrative (tradfi demand).
The only variation is the audience: “floodgates” vs. “retirement funds”—not a distinct mechanism.

❌ Few-Shot Example: ETF Buzz + Altcoin Rally
Issue: Both tweets emerge from the same underlying news event (ETF sentiment) and have the same actor class (market participants).

SIGNAL_1: Bitcoin ETF buzz reshaping institutional sentiment and capital flows
TONE_REASONING_1: ETF filings from BlackRock and others represent a legitimization moment for Bitcoin. This macro alignment deserves an institutional tone to reflect capital flow implications.
TONE_1: institutional
TWEET_1: The Bitcoin ETF wave is here! 🌊 With BlackRock's filing and fee cuts, market sentiment is buzzing. Institutional interest is aligning—this could reshape capital flows into $BTC. 👔 #Bitcoin #BTC #ETF

SIGNAL_2: [None identified — insufficient distinct signals]
TONE_REASONING_2: [N/A]
TONE_2: [N/A]
TWEET_2: [N/A]

VALID_TWEET_PAIR: false

💬 Why it fails: The second tweet (about altcoins/ETH pumping from ETF hype) doesn't qualify as a distinct Bitcoin-relevant signal. It's a side effect of the same primary narrative.
❌ Few-Shot Example: Binance.US Zero-Fee BTC
Issue: Both tweets discuss the same actor (Binance.US), same mechanism (zero-fee trading), and same likely trader reaction (volume surge).

SIGNAL_1: Binance.US introduces zero-fee trading on BTC to increase market share
TONE_REASONING_1: This is a strategic move by a major exchange to attract users and liquidity. An institutional tone captures the intent behind the policy shift and its market implications.
TONE_1: institutional
TWEET_1: Binance.US is going zero-fee on $BTC trading! 📉 Aiming to boost user sentiment and volume amid tough market conditions. This is a calculated move to capture market share—watch for increased trading activity! 👔 #Bitcoin #BTC #Binance

SIGNAL_2: [None identified — insufficient distinct signals]
TONE_REASONING_2: [N/A]
TONE_2: [N/A]
TWEET_2: [N/A]

VALID_TWEET_PAIR: false

💬 Why it fails: The idea that “other exchanges might follow” doesn’t constitute a second independent signal — it’s a continuation of the first.

ARTICLE:
\"\"\"{article_text}\"\"\"

REASONING:
\"\"\"{reasoning}\"\"\"
"""

# === FILES ===
input_file = "noticias_filtradas_bloque_7.csv"
output_file = "tweets_dataset_v.csv"


# === PROCESSING LOOP ===
df = pd.read_csv(input_file)
tweet_data = []

SAVE_INTERVAL = 600  # save every 600 tweets
save_count = 0

for idx, row in df.iterrows():
    article = row.get("article_text", "")
    reasoning = row.get("REASONING", "")
    target = row.get("target", None)

    if not article or not reasoning:
        continue

    prompt = build_prompt(article, reasoning)

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        output = response.choices[0].message.content

        # === Parse YAML manually ===
        lines = output.splitlines()
        tweet_1, tweet_2 = "", ""
        valid_pair = False

        for line in lines:
            line = line.strip()
            if line.startswith("TWEET_1:"):
                tweet_1 = line.split("TWEET_1:")[1].strip()
            elif line.startswith("TWEET_2:"):
                tweet_2 = line.split("TWEET_2:")[1].strip()
            elif line.startswith("VALID_TWEET_PAIR:"):
                valid_pair = line.split("VALID_TWEET_PAIR:")[1].strip().lower() == "true"

        if tweet_1:
            tweet_data.append({"tweet": tweet_1, "target": target})
            save_count += 1
        if valid_pair and tweet_2 and tweet_2 != "[N/A]":
            tweet_data.append({"tweet": tweet_2, "target": target})
            save_count += 1

        # === Periodic Save ===
        if save_count >= SAVE_INTERVAL:
            pd.DataFrame(tweet_data).to_csv(output_file, index=False)
            print(f"💾 Auto-saved {len(tweet_data)} tweets to {output_file}")
            save_count = 0  # reset counter

        #print(f"✅ Processed row {idx + 1}/{len(df)}")

    except Exception as e:
        print(f"⚠️ Error on row {idx + 1}: {e}")
        time.sleep(2)
        continue

# === Final Save ===
pd.DataFrame(tweet_data).to_csv(output_file, index=False)
print(f"\n✅ Done. Saved {len(tweet_data)} rows to {output_file}")
