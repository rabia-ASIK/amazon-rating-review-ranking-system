# Amazon Rating & Review Ranking System

This project focuses on improving product rating calculation and review ranking in an e-commerce environment using statistical and data-driven approaches.

## Project Goal

The goal is to:

* Calculate a more realistic product rating by considering review recency
* Rank customer reviews in a statistically reliable way
* Reduce bias caused by misleading low-vote reviews

## Methods Used

* **Time-Based Weighted Average Rating**
* **Wilson Lower Bound (WLB)** for trustworthy review ranking
* Feature engineering for review evaluation

## Tech Stack

* Python
* pandas
* numpy
* scipy

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the script:

```bash
python amazon_rating_review_ranking.py
```

## Results

* Average Rating: **4.5876**
* Time-Based Weighted Rating: **4.5956**
* Top 20 reviews selected using **Wilson Lower Bound**

## Project Structure

```
amazon_rating_review_ranking.py  -> Main project script
requirements.txt                 -> Required libraries
README.md                        -> Project documentation
```

## Why This Project Matters

Simple averages can be misleading in e-commerce systems.

This project demonstrates how statistical thinking and measurement methods can improve product evaluation, increase trust, and support better decision-making.

## Sample Output (Summary)

================== TASK 1 ==================

Average Rating: 4.587589
Time-Based Weighted Rating (28/26/24/22): 4.595593
Alternative Scenario (50/25/15/10): 4.637306

Interpretation:

Time-based weighted rating is slightly higher than the simple average.

This indicates that recent reviews are more positive.

The product’s current perception is slightly better than historical ratings.

================== TASK 2 ==================

Top 20 reviews selected using Wilson Lower Bound (WLB).

Key Insight:

Reviews are ranked using statistical confidence.

Ranking favors reviews with both high helpfulness and enough votes.

WLB range in Top 20: 0.566 – 0.958

## Author

Rabia Aşık
Junior Data Scientist | Product Analytics & Data Analysis

