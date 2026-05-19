"""Generate a synthetic dataset that exercises the EDA workflow."""

import random
from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "eda_showcase.csv"
SEED = 20260519


def repeated_values(
    counts: dict[str, int],
    rng: random.Random,
) -> list[str]:
    """Create a shuffled list from category counts."""
    values = [
        value
        for value, count in counts.items()
        for _ in range(count)
    ]
    rng.shuffle(values)
    return values


def clipped(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Constrain a numeric value to a fixed range."""
    return min(max(value, minimum), maximum)


def build_showcase_dataset() -> pd.DataFrame:
    """Build a deterministic cafe operations dataset with known EDA signals."""
    rng = random.Random(SEED)
    row_count = 120

    customer_segments = repeated_values(
        {
            "Standard": 44,
            "Budget": 36,
            "Premium": 36,
            "Pilot": 4,
        },
        rng,
    )
    channels = repeated_values(
        {
            "In Store": 45,
            "Delivery": 35,
            "Mobile App": 30,
            "Catering": 10,
        },
        rng,
    )
    regions = repeated_values(
        {
            "North": 34,
            "South": 31,
            "East": 29,
            "West": 26,
        },
        rng,
    )
    product_categories = repeated_values(
        {
            "Coffee": 36,
            "Bakery": 30,
            "Lunch": 28,
            "Tea": 18,
            "Merchandise": 8,
        },
        rng,
    )
    promo_types = repeated_values(
        {
            "No Promo": 48,
            "Loyalty": 32,
            "Seasonal": 24,
            "New Customer": 12,
            "Influencer": 4,
        },
        rng,
    )

    segment_spend = {
        "Budget": 22,
        "Standard": 48,
        "Premium": 105,
        "Pilot": 14,
    }
    channel_wait = {
        "In Store": 8,
        "Delivery": 28,
        "Mobile App": 5,
        "Catering": 18,
    }
    product_spend = {
        "Coffee": 2,
        "Bakery": 4,
        "Lunch": 10,
        "Tea": 1,
        "Merchandise": 14,
    }

    rows = []
    for index in range(row_count):
        segment = customer_segments[index]
        channel = channels[index]
        product_category = product_categories[index]

        monthly_spend = (
            segment_spend[segment]
            + product_spend[product_category]
            + rng.uniform(-5, 5)
        )
        monthly_spend = round(max(monthly_spend, 5), 2)

        visit_count = max(1, round(monthly_spend / 8 + rng.uniform(-1, 1)))
        lifetime_value = round(monthly_spend * (visit_count + 6) + monthly_spend ** 1.2, 2)

        wait_time_minutes = round(channel_wait[channel] + rng.uniform(-3, 3), 2)
        satisfaction_score = round(
            clipped(98 - wait_time_minutes * 2.15 + rng.uniform(-3, 3), 25, 100),
            2,
        )

        discount_rate = round(rng.uniform(0.02, 0.35), 2)
        if index % 20 == 0:
            discount_rate = None

        sparse_quality_score = round(rng.uniform(55, 98), 2)
        if index % 3 == 0:
            sparse_quality_score = None

        rows.append({
            "customer_id": f"CUST-{index + 1:04d}",
            "region": regions[index],
            "channel": channel,
            "customer_segment": segment,
            "product_category": product_category,
            "promo_type": promo_types[index],
            "monthly_spend": monthly_spend,
            "visit_count": visit_count,
            "lifetime_value": lifetime_value,
            "wait_time_minutes": wait_time_minutes,
            "satisfaction_score": satisfaction_score,
            "discount_rate": discount_rate,
            "sparse_quality_score": sparse_quality_score,
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and save the showcase dataset."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_showcase_dataset().to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
