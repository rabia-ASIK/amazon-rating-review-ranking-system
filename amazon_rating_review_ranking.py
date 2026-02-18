"""
============================================================
PROJECT: Rating Product & Sorting Reviews in Amazon
AUTHOR: Rabia (your presentation-ready version)
============================================================

🎯 Business Problem
------------------
E-commerce platforms face two critical measurement problems:

1) Product Rating Problem:
   - A simple average rating treats old and new reviews equally.
   - But product quality and customer expectations change over time.
   - We want a rating that reflects the CURRENT customer experience.

2) Review Sorting Problem:
   - Sorting by raw helpful votes or raw ratios can be misleading.
   - Low-vote reviews can look "perfect" by chance or manipulation.
   - We want the most RELIABLE reviews to show on the product page.

✅ This script solves both problems:
   Task 1: Time-based weighted product rating
   Task 2: Statistical review ranking with Wilson Lower Bound (WLB)

------------------------------------------------------------
HOW TO RUN (Local)
------------------
1) Put the dataset file in the same folder as this script:
   amazon_review.csv

2) Install dependencies:
   pip install pandas numpy scipy

3) Run:
   python amazon_rating_review_ranking.py

------------------------------------------------------------
NOTE FOR GITHUB
---------------
- Do NOT hardcode Windows paths in GitHub projects.
- Use a relative path ("amazon_review.csv") and provide a clear error if missing.
============================================================
"""

# ==============================
# 0) Imports & Display Settings
# ==============================
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


# ==============================
# 1) Helper: Safe CSV Loading
# ==============================
def load_data(file_path: str = "amazon_review.csv") -> pd.DataFrame:
    """
    Loads the dataset safely.

    Why we do this:
    - In GitHub projects, absolute paths (D:\\...) won't work for other users.
    - So we use relative paths and a clear error message.

    Parameters
    ----------
    file_path : str
        CSV file name or relative path.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: '{file_path}'.\n"
            "✅ Put 'amazon_review.csv' in the SAME folder as this script.\n"
            "✅ Or change file_path to the correct location."
        )
    return pd.read_csv(file_path)


# ==========================================================
# TASK 1) Product Rating
# ==========================================================
def time_based_weighted_average(
    dataframe: pd.DataFrame,
    w1: int = 28,
    w2: int = 26,
    w3: int = 24,
    w4: int = 22,
    time_col: str = "day_diff",
    rating_col: str = "overall",
) -> float:
    """
    Calculates time-based weighted average rating using quantile-based segmentation.

    Why quantiles?
    - Instead of choosing arbitrary day ranges (e.g., last 30 days),
      quantiles split the data into balanced groups.
    - This is more stable across datasets of different time distributions.

    Why weights?
    - Newer reviews better reflect current customer experience.
    - We assign higher weight to newer segments.

    w1..w4 are percentages and should sum to 100.

    Returns
    -------
    float
        Time-based weighted average rating.
    """
    # Quantile thresholds (25%, 50%, 75%)
    q1 = dataframe[time_col].quantile(0.25)
    q2 = dataframe[time_col].quantile(0.50)
    q3 = dataframe[time_col].quantile(0.75)

    # Weighted sum of segment means
    return (
        dataframe.loc[dataframe[time_col] <= q1, rating_col].mean() * w1 / 100
        + dataframe.loc[(dataframe[time_col] > q1) & (dataframe[time_col] <= q2), rating_col].mean() * w2 / 100
        + dataframe.loc[(dataframe[time_col] > q2) & (dataframe[time_col] <= q3), rating_col].mean() * w3 / 100
        + dataframe.loc[dataframe[time_col] > q3, rating_col].mean() * w4 / 100
    )


def explain_task1_results(avg_rating: float, tw_rating: float) -> str:
    """
    Returns a presentation-friendly interpretation of Task 1 results.
    """
    if tw_rating > avg_rating:
        return (
            "Time-based weighted rating is slightly HIGHER than the simple average.\n"
            "➡️ This indicates that more recent reviews are more positive.\n"
            "➡️ The product's CURRENT perception seems slightly better than the historical average."
        )
    elif tw_rating < avg_rating:
        return (
            "Time-based weighted rating is slightly LOWER than the simple average.\n"
            "➡️ This indicates that more recent reviews are more negative.\n"
            "➡️ The product's CURRENT perception may be worse than the historical average."
        )
    return (
        "Time-based weighted rating is VERY close to the simple average.\n"
        "➡️ Recent reviews are consistent with the historical trend."
    )


# ==========================================================
# TASK 2) Review Ranking
# ==========================================================
def score_pos_neg_diff(up: int, down: int) -> int:
    """
    Net helpfulness score: up - down

    Why?
    - Simple measure of net positive feedback.
    - But it ignores the total vote size and statistical confidence.
    """
    return up - down


def score_average_rating(up: int, down: int) -> float:
    """
    Helpful ratio: up / (up + down)

    Why?
    - Measures the proportion of helpful votes.
    - Limitation: low-vote reviews can look perfect by chance (e.g., 1/1 = 100%).
    """
    if up + down == 0:
        return 0.0
    return up / (up + down)


def wilson_lower_bound(up: int, down: int, confidence: float = 0.95) -> float:
    """
    Wilson Lower Bound (WLB) score for Bernoulli parameter.

    Why WLB?
    - It ranks reviews by a statistically safe lower bound of the helpful ratio.
    - It penalizes low sample size (few votes).
    - This prevents 'lucky' or manipulated reviews from dominating rankings.

    Key Interview Explanation:
    - A review with 1 helpful vote and 0 unhelpful votes has ratio=1.0,
      but WLB will be much lower because confidence is low.
    """
    n = up + down
    if n == 0:
        return 0.0

    z = norm.ppf(1 - (1 - confidence) / 2)  # ~1.96 for 95%
    phat = up / n

    return (phat + z * z / (2 * n) - z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def add_review_scores(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds helpful_no and ranking scores to the dataframe.

    Steps:
    1) helpful_no = total_vote - helpful_yes
    2) score_pos_neg_diff
    3) score_average_rating
    4) wilson_lower_bound

    Why apply(axis=1)?
    - We compute row-level scores using helpful_yes/helpful_no for each review.

    Returns
    -------
    pd.DataFrame
        Dataframe with new score columns.
    """
    df = dataframe.copy()

    # Feature Engineering: helpful_no
    df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

    # Scoring
    df["score_pos_neg_diff"] = df.apply(
        lambda x: score_pos_neg_diff(int(x["helpful_yes"]), int(x["helpful_no"])),
        axis=1
    )
    df["score_average_rating"] = df.apply(
        lambda x: score_average_rating(int(x["helpful_yes"]), int(x["helpful_no"])),
        axis=1
    )
    df["wilson_lower_bound"] = df.apply(
        lambda x: wilson_lower_bound(int(x["helpful_yes"]), int(x["helpful_no"])),
        axis=1
    )

    return df


def get_top_n_reviews(dataframe: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Returns top N reviews by Wilson Lower Bound.

    Why WLB sorting?
    - More reliable than raw vote counts or simple ratios.

    Returns a clean view for presentation.
    """
    cols = ["reviewText", "helpful_yes", "total_vote", "wilson_lower_bound"]
    return dataframe.sort_values("wilson_lower_bound", ascending=False).head(n)[cols]


def explain_task2_results(top_reviews: pd.DataFrame) -> str:
    """
    Presentation-friendly interpretation of Task 2 results.
    """
    max_wlb = top_reviews["wilson_lower_bound"].max()
    min_wlb = top_reviews["wilson_lower_bound"].min()

    return (
        "Top 20 reviews were selected using Wilson Lower Bound.\n"
        "➡️ This ranking favors reviews with BOTH high helpfulness and high confidence (enough votes).\n"
        f"➡️ WLB range in Top 20: {min_wlb:.3f} to {max_wlb:.3f}\n"
        "➡️ This approach reduces the risk of misleading low-vote reviews appearing at the top."
    )


# ==========================================================
# MAIN (Run everything)
# ==========================================================
if __name__ == "__main__":
    print("\n================== PROJECT START ==================\n")

    # 1) Load data
    df = load_data("amazon_review.csv")

    print("===== DATA CHECK =====")
    print("Shape:", df.shape)
    print(df.head(3))
    print()

    # --------------------------
    # TASK 1: Product rating
    # --------------------------
    print("================== TASK 1 ==================")
    avg_rating = df["overall"].mean()

    # Primary (your balanced weights)
    tw_rating = time_based_weighted_average(df, w1=28, w2=26, w3=24, w4=22)

    # Optional (mentor-like aggressive scenario) -> for interview Q: "what if we care much more about recency?"
    tw_rating_aggressive = time_based_weighted_average(df, w1=50, w2=25, w3=15, w4=10)

    print(f"Average Rating: {avg_rating:.6f}")
    print(f"Time-Based Weighted Rating (28/26/24/22): {tw_rating:.6f}")
    print(f"Alternative Scenario (50/25/15/10): {tw_rating_aggressive:.6f}\n")

    print("Task 1 Interpretation:")
    print(explain_task1_results(avg_rating, tw_rating))
    print()

    # --------------------------
    # TASK 2: Review ranking
    # --------------------------
    print("================== TASK 2 ==================")
    df_scored = add_review_scores(df)

    print("Score columns check (first 10 rows):")
    print(df_scored[["helpful_yes", "helpful_no", "total_vote", "score_pos_neg_diff", "score_average_rating", "wilson_lower_bound"]].head(10))
    print()

    top_20 = get_top_n_reviews(df_scored, n=20)
    print("Top 20 Reviews (by Wilson Lower Bound):")
    print(top_20)
    print()

    print("Task 2 Interpretation:")
    print(explain_task2_results(top_20))
    print("\n================== PROJECT END ==================\n")

"""
============================================================
INTERVIEW Q&A (Quick Answers You Can Say)
============================================================

Q1) Why not use simple average rating?
- Because it ignores recency. Old and new reviews have equal impact, which can misrepresent current product quality.

Q2) Why quantiles for time segments?
- Quantiles create balanced groups based on the data distribution, making the method stable across different datasets.

Q3) Why two weight scenarios?
- 28/26/24/22 is a balanced approach.
- 50/25/15/10 is an alternative when recency is extremely important.
  It’s useful to show sensitivity analysis and business-driven decisions.

Q4) Why not rank reviews by helpful_yes or helpful ratio?
- helpful_yes favors older reviews (more time to collect votes).
- ratio can be misleading for low-vote reviews (1/1 looks perfect).

Q5) Why Wilson Lower Bound?
- It incorporates confidence by penalizing small sample sizes, producing more reliable rankings.

============================================================
"""
