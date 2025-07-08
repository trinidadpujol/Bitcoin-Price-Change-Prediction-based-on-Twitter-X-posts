
import asyncio
import pandas as pd
from twscrape import API

async def scrape_tweet_account_jun_jul_2025():
    """
    Scrapes tweets from the user between June 1, 2025 and July 31, 2025 (inclusive)
    and saves them to a CSV.
    """
    api = API()
    await api.pool.add_account(
        username="user_name",            # ← Replace with your account username
        password="password",              # ← Replace with your account password
        email="email",                   # ← Replace with your account email
        email_password="email_password", # ← Replace with your email password
        cookies=(
            "auth_token=AAA; "           # ← Replace with your account authentication token
            "ct0=BBB"                    # ← Replace with your account session token
            "guest_id=CCC"               # ← Replace with your account guest ID
        )
    )
    await api.pool.login_all()

    user_id = 3005014565          # ← Replace with the account you want to scrape
    tweets  = []

    async for tweet in api.user_tweets(user_id, limit=2000):
        y, m = tweet.date.year, tweet.date.month

        if y == 2025 and m in (6, 7):
            tweets.append(tweet.dict())

        # If the timeline has already passed July 2025,
        # we can break to save API calls.
        if y < 2025 or (y == 2025 and m < 6):
            break

    df = pd.DataFrame(tweets)
    df.to_csv("bitcoin_news_jun_jul_2025.csv", index=False, encoding="utf-8")
    print(f"Tweets in Jun-Jul 2025: {len(df)}")

if __name__ == "__main__":
    asyncio.run(scrape_tweet_account_jun_jul_2025())


