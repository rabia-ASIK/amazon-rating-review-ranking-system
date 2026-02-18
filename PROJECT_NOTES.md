# Project Notes — Amazon Rating & Review Ranking

## Problem
Simple average ratings and naive review sorting methods can be misleading in e-commerce systems.

## Key Decisions
- Used time-based weighting to prioritize recent reviews.
- Applied quantile segmentation to split review periods fairly.
- Used Wilson Lower Bound (WLB) to rank reviews by statistical reliability.

## Why Wilson Lower Bound?
Simple helpful ratios can be misleading when vote counts are low.
WLB provides a safer lower confidence bound and reduces ranking bias.

## My Learning
This project helped me understand how small analytical decisions can directly impact product perception and user trust.
