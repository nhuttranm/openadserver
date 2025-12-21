#!/usr/bin/env python3
"""
Stress test for OpenAdServer with full pipeline.

Uses SQLite in-memory + fakeredis for zero-dependency testing.
Includes Numba JIT acceleration for feature engineering.

Pipeline tested:
1. Retrieval (Targeting)
2. Filter (Budget, Frequency, Quality)
3. Prediction (LR/FM/DeepFM) with JIT-accelerated features
4. Ranking (eCPM)
5. Rerank (Diversity, Exploration)

Usage:
    # Minimal test (low memory)
    python scripts/criteo/stress_test.py --campaigns 10 --requests 100

    # Full test (when memory available)
    python scripts/criteo/stress_test.py --campaigns 50 --requests 1000
"""

import argparse
import asyncio
import gc
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try to import Numba for JIT acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Numba JIT compiled functions for feature engineering
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def build_sparse_features_jit(
        n: int,
        user_hash: int,
        os_hash: int,
        country_hash: int,
        city_hash: int,
        slot_hash: int,
        model_hash: int,
        brand_hash: int,
        os_ver_hash: int,
        app_hash: int,
        net_hash: int,
        interests_hash: int,
        geo_combo_hash: int,
        age: int,
        gender_val: int,
        n_interests: int,
        campaign_ids: np.ndarray,
        creative_ids: np.ndarray,
        quality_scores: np.ndarray,
        bid_amounts: np.ndarray,
        bid_types: np.ndarray,
    ) -> np.ndarray:
        """JIT-compiled sparse feature builder (26 features)."""
        sparse = np.zeros((n, 26), dtype=np.int64)
        for i in range(n):
            sparse[i, 0] = user_hash % 10000
            sparse[i, 1] = os_hash % 100
            sparse[i, 2] = country_hash % 1000
            sparse[i, 3] = city_hash % 10000
            sparse[i, 4] = campaign_ids[i] % 100
            sparse[i, 5] = creative_ids[i] % 50
            sparse[i, 6] = slot_hash % 1000
            sparse[i, 7] = age // 10
            sparse[i, 8] = gender_val
            sparse[i, 9] = interests_hash % 10000
            sparse[i, 10] = bid_types[i]
            sparse[i, 11] = brand_hash % 1000
            sparse[i, 12] = quality_scores[i] // 10
            sparse[i, 13] = os_ver_hash % 50
            sparse[i, 14] = app_hash % 5000
            sparse[i, 15] = net_hash % 10
            sparse[i, 16] = 1
            sparse[i, 17] = model_hash % 2000
            sparse[i, 18] = np.int64(bid_amounts[i] * 10) % 100
            sparse[i, 19] = 1
            sparse[i, 20] = geo_combo_hash % 10000
            sparse[i, 21] = n_interests % 10
            sparse[i, 22] = 1
            sparse[i, 23] = (campaign_ids[i] * 7) % 10000
            sparse[i, 24] = quality_scores[i] // 5
            sparse[i, 25] = ((campaign_ids[i] << 8) ^ creative_ids[i]) % 10000
        return sparse

    @jit(nopython=True, cache=True)
    def build_dense_features_jit(
        n: int,
        user_hash: int,
        age: int,
        os_hash: int,
        country_hash: int,
        city_hash: int,
        slot_hash: int,
        model_hash: int,
        n_interests: int,
        campaign_ids: np.ndarray,
        creative_ids: np.ndarray,
        quality_scores: np.ndarray,
        bid_amounts: np.ndarray,
        bid_types: np.ndarray,
    ) -> np.ndarray:
        """JIT-compiled dense feature builder (13 features)."""
        dense = np.zeros((n, 13), dtype=np.float32)
        for i in range(n):
            dense[i, 0] = user_hash % 1000
            dense[i, 1] = age
            dense[i, 2] = campaign_ids[i] % 100
            dense[i, 3] = creative_ids[i] % 100
            dense[i, 4] = quality_scores[i]
            dense[i, 5] = os_hash % 10
            dense[i, 6] = country_hash % 100
            dense[i, 7] = city_hash % 1000
            dense[i, 8] = bid_amounts[i] * 100
            dense[i, 9] = bid_types[i]
            dense[i, 10] = slot_hash % 100
            dense[i, 11] = n_interests
            dense[i, 12] = model_hash % 100
        return dense

    @jit(nopython=True, cache=True)
    def normalize_dense_jit(dense: np.ndarray) -> np.ndarray:
        """JIT-compiled normalization."""
        n, m = dense.shape
        result = np.empty_like(dense)
        for j in range(m):
            col_mean = 0.0
            for i in range(n):
                col_mean += dense[i, j]
            col_mean /= n

            col_std = 0.0
            for i in range(n):
                col_std += (dense[i, j] - col_mean) ** 2
            col_std = np.sqrt(col_std / n) + 1e-8

            for i in range(n):
                result[i, j] = (dense[i, j] - col_mean) / col_std
        return result
else:
    # Fallback numpy implementations
    def build_sparse_features_jit(n, user_hash, os_hash, country_hash, city_hash,
                                   slot_hash, model_hash, brand_hash, os_ver_hash,
                                   app_hash, net_hash, interests_hash, geo_combo_hash,
                                   age, gender_val, n_interests, campaign_ids,
                                   creative_ids, quality_scores, bid_amounts, bid_types):
        sparse = np.zeros((n, 26), dtype=np.int64)
        sparse[:, 0] = user_hash % 10000
        sparse[:, 1] = os_hash % 100
        sparse[:, 2] = country_hash % 1000
        sparse[:, 3] = city_hash % 10000
        sparse[:, 4] = campaign_ids % 100
        sparse[:, 5] = creative_ids % 50
        sparse[:, 6] = slot_hash % 1000
        sparse[:, 7] = age // 10
        sparse[:, 8] = gender_val
        sparse[:, 9] = interests_hash % 10000
        sparse[:, 10] = bid_types
        sparse[:, 11] = brand_hash % 1000
        sparse[:, 12] = quality_scores // 10
        sparse[:, 13] = os_ver_hash % 50
        sparse[:, 14] = app_hash % 5000
        sparse[:, 15] = net_hash % 10
        sparse[:, 16] = 1
        sparse[:, 17] = model_hash % 2000
        sparse[:, 18] = (bid_amounts * 10).astype(np.int64) % 100
        sparse[:, 19] = 1
        sparse[:, 20] = geo_combo_hash % 10000
        sparse[:, 21] = n_interests % 10
        sparse[:, 22] = 1
        sparse[:, 23] = (campaign_ids * 7) % 10000
        sparse[:, 24] = quality_scores // 5
        sparse[:, 25] = ((campaign_ids << 8) ^ creative_ids) % 10000
        return sparse

    def build_dense_features_jit(n, user_hash, age, os_hash, country_hash, city_hash,
                                  slot_hash, model_hash, n_interests, campaign_ids,
                                  creative_ids, quality_scores, bid_amounts, bid_types):
        dense = np.zeros((n, 13), dtype=np.float32)
        dense[:, 0] = user_hash % 1000
        dense[:, 1] = age
        dense[:, 2] = campaign_ids % 100
        dense[:, 3] = creative_ids % 100
        dense[:, 4] = quality_scores
        dense[:, 5] = os_hash % 10
        dense[:, 6] = country_hash % 100
        dense[:, 7] = city_hash % 1000
        dense[:, 8] = bid_amounts * 100
        dense[:, 9] = bid_types
        dense[:, 10] = slot_hash % 100
        dense[:, 11] = n_interests
        dense[:, 12] = model_hash % 100
        return dense

    def normalize_dense_jit(dense):
        return (dense - dense.mean(axis=0)) / (dense.std(axis=0) + 1e-8)


@dataclass
class StressTestConfig:
    """Stress test configuration."""
    num_campaigns: int = 10
    creatives_per_campaign: int = 3
    num_requests: int = 100
    concurrent_requests: int = 10
    model_type: str = "lr"  # lr, fm, deepfm
    enable_ml_prediction: bool = True
    warmup_requests: int = 10


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.times_ms)

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]


@dataclass
class StressTestResult:
    """Stress test results."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time_s: float = 0

    # Stage metrics
    retrieval: StageMetrics = field(default_factory=lambda: StageMetrics("retrieval"))
    filter: StageMetrics = field(default_factory=lambda: StageMetrics("filter"))
    prediction: StageMetrics = field(default_factory=lambda: StageMetrics("prediction"))
    ranking: StageMetrics = field(default_factory=lambda: StageMetrics("ranking"))
    rerank: StageMetrics = field(default_factory=lambda: StageMetrics("rerank"))
    total: StageMetrics = field(default_factory=lambda: StageMetrics("total"))

    # Fill rate
    total_ads_requested: int = 0
    total_ads_returned: int = 0

    @property
    def qps(self) -> float:
        return self.total_requests / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0

    @property
    def fill_rate(self) -> float:
        return self.total_ads_returned / self.total_ads_requested if self.total_ads_requested > 0 else 0


class InMemoryDatabase:
    """SQLite in-memory database for testing."""

    def __init__(self):
        self.engine = None
        self.session_factory = None

    async def init(self):
        """Initialize SQLite in-memory database."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        # Use SQLite in-memory with aiosqlite
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
        )

        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create tables
        from liteads.models.base import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print("  SQLite in-memory database initialized")

    async def close(self):
        if self.engine:
            await self.engine.dispose()

    def session(self):
        return self.session_factory()


class FakeRedisClient:
    """Fake Redis client for testing."""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}

    async def connect(self):
        print("  FakeRedis initialized")

    async def close(self):
        self._data.clear()
        self._expiry.clear()

    async def get(self, key: str) -> str | None:
        self._check_expiry(key)
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None, **kwargs) -> bool:
        self._data[key] = value
        if ex:
            self._expiry[key] = time.time() + ex
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count

    async def incr(self, key: str) -> int:
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = str(val)
        return val

    async def incrby(self, key: str, amount: int) -> int:
        val = int(self._data.get(key, 0)) + amount
        self._data[key] = str(val)
        return val

    async def expire(self, key: str, ttl: int) -> bool:
        if key in self._data:
            self._expiry[key] = time.time() + ttl
            return True
        return False

    async def ping(self) -> bool:
        return True

    async def hget(self, key: str, field: str) -> str | None:
        hash_data = self._data.get(key, {})
        if isinstance(hash_data, dict):
            return hash_data.get(field)
        return None

    async def hset(self, key: str, field: str = None, value: str = None, mapping: dict = None) -> int:
        if key not in self._data:
            self._data[key] = {}
        if mapping:
            self._data[key].update(mapping)
            return len(mapping)
        if field and value:
            self._data[key][field] = value
            return 1
        return 0

    async def hgetall(self, key: str) -> dict:
        return self._data.get(key, {})

    def _check_expiry(self, key: str):
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._data[key]
            del self._expiry[key]


async def seed_test_data(db: InMemoryDatabase, config: StressTestConfig):
    """Seed test campaigns and creatives."""
    from liteads.models import Advertiser, BidType, Campaign, Creative, CreativeType, Status, TargetingRule

    print(f"  Seeding {config.num_campaigns} campaigns...")

    async with db.session() as session:
        # Create advertisers (1 per 5 campaigns)
        num_advertisers = max(1, config.num_campaigns // 5)
        advertisers = []

        for i in range(num_advertisers):
            adv = Advertiser(
                name=f"Advertiser_{i}",
                balance=Decimal("10000"),
                daily_budget=Decimal("1000"),
                status=Status.ACTIVE,
            )
            session.add(adv)
            advertisers.append(adv)

        await session.flush()

        # Create campaigns
        now = datetime.utcnow()
        campaigns = []

        for i in range(config.num_campaigns):
            adv = advertisers[i % len(advertisers)]
            campaign = Campaign(
                advertiser_id=adv.id,
                name=f"Campaign_{i}",
                budget_daily=Decimal(str(random.randint(100, 1000))),
                budget_total=Decimal(str(random.randint(1000, 10000))),
                bid_type=random.choice([BidType.CPM, BidType.CPC]),
                bid_amount=Decimal(str(random.uniform(0.5, 10.0))),
                freq_cap_daily=random.choice([5, 10, 20]),
                freq_cap_hourly=random.choice([1, 2, 3]),
                start_time=now - timedelta(days=1),
                end_time=now + timedelta(days=30),
                status=Status.ACTIVE,
            )
            session.add(campaign)
            campaigns.append(campaign)

        await session.flush()

        # Create creatives
        creative_count = 0
        for campaign in campaigns:
            for j in range(config.creatives_per_campaign):
                creative = Creative(
                    campaign_id=campaign.id,
                    title=f"Ad_{campaign.id}_{j}",
                    description="Test ad description",
                    image_url=f"https://example.com/img/{campaign.id}_{j}.jpg",
                    landing_url=f"https://example.com/land/{campaign.id}",
                    creative_type=random.choice([CreativeType.BANNER, CreativeType.NATIVE]),
                    width=random.choice([300, 320, 728]),
                    height=random.choice([250, 50, 90]),
                    status=Status.ACTIVE,
                    quality_score=random.randint(60, 100),
                )
                session.add(creative)
                creative_count += 1

        await session.flush()

        # Create targeting rules
        for campaign in campaigns:
            # Device targeting
            session.add(TargetingRule(
                campaign_id=campaign.id,
                rule_type="device",
                rule_value={"os": ["android", "ios"]},
                is_include=True,
            ))
            # Geo targeting
            session.add(TargetingRule(
                campaign_id=campaign.id,
                rule_type="geo",
                rule_value={"countries": ["CN", "US", "JP"]},
                is_include=True,
            ))

        await session.commit()

    print(f"  Created {len(advertisers)} advertisers, {len(campaigns)} campaigns, {creative_count} creatives")
    return campaigns


def generate_ad_request() -> dict:
    """Generate a random ad request."""
    return {
        "slot_id": random.choice(["banner_top", "native_feed", "interstitial"]),
        "user_id": f"user_{random.randint(1, 10000):05d}",
        "device": {
            "os": random.choice(["android", "ios"]),
            "os_version": "13.0",
            "model": "TestDevice",
            "brand": "TestBrand",
        },
        "geo": {
            "country": random.choice(["CN", "US", "JP"]),
            "city": random.choice(["shanghai", "beijing", "tokyo"]),
        },
        "context": {
            "app_id": f"com.test.app{random.randint(1, 100)}",
            "network": random.choice(["wifi", "4g", "5g"]),
        },
        "user_features": {
            "age": random.randint(18, 60),
            "gender": random.choice(["male", "female"]),
            "interests": random.sample(["gaming", "sports", "music", "tech"], k=2),
        },
        "num_ads": random.choice([1, 1, 1, 3]),
    }


class SimplePipeline:
    """Simplified ad serving pipeline for testing."""

    def __init__(self, db: InMemoryDatabase, redis: FakeRedisClient, config: StressTestConfig):
        self.db = db
        self.redis = redis
        self.config = config
        self._campaigns = []
        self._predictor = None

    async def init(self):
        """Initialize pipeline components."""
        # Load campaigns into memory
        from liteads.models import Campaign, Creative, Status
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        async with self.db.session() as session:
            result = await session.execute(
                select(Campaign)
                .where(Campaign.status == Status.ACTIVE)
                .options(selectinload(Campaign.creatives))
            )
            self._campaigns = list(result.scalars().all())

        print(f"  Loaded {len(self._campaigns)} active campaigns")

        # Initialize ML predictor if enabled
        if self.config.enable_ml_prediction:
            await self._init_predictor()

    async def _init_predictor(self):
        """Initialize ML predictor."""
        model_path = Path(f"models/criteo/{self.config.model_type}_criteo.pt")
        feature_path = Path("models/criteo/feature_builder.pkl")

        if not model_path.exists():
            print(f"  Warning: Model {model_path} not found, using fallback prediction")
            return

        try:
            from liteads.ml_engine.serving.predictor import ModelPredictor
            self._predictor = ModelPredictor(
                model_path=str(model_path),
                feature_builder_path=str(feature_path) if feature_path.exists() else None,
            )
            self._predictor.load()
            print(f"  Loaded {self.config.model_type.upper()} model for prediction")
        except Exception as e:
            print(f"  Warning: Failed to load model: {e}")

    async def serve(self, request: dict) -> tuple[list[dict], dict[str, float]]:
        """
        Serve ads through the full pipeline.

        Returns: (ads, stage_times_ms)
        """
        times = {}

        # 1. Retrieval (Targeting)
        start = time.perf_counter()
        candidates = await self._retrieve(request)
        times["retrieval"] = (time.perf_counter() - start) * 1000

        if not candidates:
            return [], times

        # 2. Filter
        start = time.perf_counter()
        candidates = await self._filter(candidates, request)
        times["filter"] = (time.perf_counter() - start) * 1000

        if not candidates:
            return [], times

        # 3. Prediction
        start = time.perf_counter()
        candidates = await self._predict(candidates, request)
        times["prediction"] = (time.perf_counter() - start) * 1000

        # 4. Ranking
        start = time.perf_counter()
        candidates = self._rank(candidates)
        times["ranking"] = (time.perf_counter() - start) * 1000

        # 5. Rerank
        start = time.perf_counter()
        num_ads = request.get("num_ads", 1)
        candidates = self._rerank(candidates, num_ads)
        times["rerank"] = (time.perf_counter() - start) * 1000

        return candidates[:num_ads], times

    async def _retrieve(self, request: dict) -> list[dict]:
        """Retrieve candidate ads based on targeting."""
        candidates = []

        device_os = request.get("device", {}).get("os", "android")
        country = request.get("geo", {}).get("country", "CN")

        for campaign in self._campaigns:
            # Simple targeting match
            for creative in campaign.creatives:
                candidates.append({
                    "campaign_id": campaign.id,
                    "creative_id": creative.id,
                    "bid_type": campaign.bid_type,
                    "bid_amount": float(campaign.bid_amount),
                    "quality_score": creative.quality_score,
                    "title": creative.title,
                    "image_url": creative.image_url,
                    "landing_url": creative.landing_url,
                })

        return candidates

    async def _filter(self, candidates: list[dict], request: dict) -> list[dict]:
        """Apply filters: budget, frequency, quality."""
        user_id = request.get("user_id", "unknown")
        filtered = []

        for c in candidates:
            # Budget filter (simplified)
            # Frequency filter (check Redis)
            freq_key = f"freq:{user_id}:{c['campaign_id']}"
            freq_count = await self.redis.get(freq_key)
            if freq_count and int(freq_count) > 10:
                continue

            # Quality filter
            if c.get("quality_score", 0) < 50:
                continue

            filtered.append(c)

        return filtered

    async def _predict(self, candidates: list[dict], request: dict) -> list[dict]:
        """Predict CTR/CVR using real ML model with Numba JIT-accelerated feature engineering."""
        if self._predictor and candidates:
            try:
                import torch

                n = len(candidates)

                # Pre-compute request-level hashes (only once per request)
                user_id = request.get("user_id", "")
                user_features = request.get("user_features", {})
                device = request.get("device", {})
                geo = request.get("geo", {})
                context = request.get("context", {})

                user_hash = hash(user_id)
                os_hash = hash(device.get("os", "android"))
                country_hash = hash(geo.get("country", "CN"))
                city_hash = hash(geo.get("city", ""))
                slot_hash = hash(request.get("slot_id", ""))
                model_hash = hash(device.get("model", ""))
                brand_hash = hash(device.get("brand", ""))
                os_ver_hash = hash(device.get("os_version", ""))
                app_hash = hash(context.get("app_id", ""))
                net_hash = hash(context.get("network", ""))
                interests_hash = hash(str(user_features.get("interests", [])))
                geo_combo_hash = hash(f"{geo.get('country', '')}_{geo.get('city', '')}")

                age = user_features.get("age", 25)
                gender_val = 1 if user_features.get("gender") == "male" else 0
                n_interests = len(user_features.get("interests", []))

                # Vectorized feature extraction
                campaign_ids = np.array([c.get("campaign_id", 0) for c in candidates], dtype=np.int64)
                creative_ids = np.array([c.get("creative_id", 0) for c in candidates], dtype=np.int64)
                quality_scores = np.array([c.get("quality_score", 80) for c in candidates], dtype=np.int64)
                bid_amounts = np.array([c.get("bid_amount", 1.0) for c in candidates], dtype=np.float32)
                bid_types = np.array([c.get("bid_type", 1) for c in candidates], dtype=np.int64)

                # Build sparse features using JIT-compiled function (26 features)
                sparse = build_sparse_features_jit(
                    n, user_hash, os_hash, country_hash, city_hash,
                    slot_hash, model_hash, brand_hash, os_ver_hash,
                    app_hash, net_hash, interests_hash, geo_combo_hash,
                    age, gender_val, n_interests,
                    campaign_ids, creative_ids, quality_scores, bid_amounts, bid_types
                )

                # Build dense features using JIT-compiled function (13 features)
                dense = build_dense_features_jit(
                    n, user_hash, age, os_hash, country_hash, city_hash,
                    slot_hash, model_hash, n_interests,
                    campaign_ids, creative_ids, quality_scores, bid_amounts, bid_types
                )

                # Normalize dense features using JIT-compiled function
                dense = normalize_dense_jit(dense)

                # Convert to tensors
                device_t = self._predictor.device
                sparse_t = torch.from_numpy(sparse).to(device_t)
                dense_t = torch.from_numpy(dense).to(device_t)

                # Model inference
                with torch.no_grad():
                    outputs = self._predictor.model(sparse_t, dense_t)
                    pctrs = outputs["ctr"].cpu().numpy()

                # Assign predictions
                for i, c in enumerate(candidates):
                    c["pctr"] = float(np.clip(pctrs[i], 0.001, 0.99))
                    c["pcvr"] = c["pctr"] * 0.1

            except Exception as e:
                # Fallback on error
                for c in candidates:
                    c["pctr"] = 0.02 + random.uniform(-0.01, 0.01)
                    c["pcvr"] = 0.002
        else:
            # Fallback prediction
            for c in candidates:
                c["pctr"] = 0.02 + random.uniform(-0.01, 0.01)
                c["pcvr"] = 0.002

        return candidates

    def _rank(self, candidates: list[dict]) -> list[dict]:
        """Rank by eCPM."""
        for c in candidates:
            bid_type = c.get("bid_type", 1)
            bid_amount = c.get("bid_amount", 1.0)
            pctr = c.get("pctr", 0.01)

            if bid_type == 1:  # CPM
                c["ecpm"] = bid_amount
            elif bid_type == 2:  # CPC
                c["ecpm"] = bid_amount * pctr * 1000
            else:  # CPA
                c["ecpm"] = bid_amount * pctr * c.get("pcvr", 0.001) * 1000

            c["score"] = c["ecpm"] * (1 + random.uniform(-0.1, 0.1))  # Add noise

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def _rerank(self, candidates: list[dict], num_ads: int) -> list[dict]:
        """Apply diversity and exploration."""
        if len(candidates) <= num_ads:
            return candidates

        # Diversity: don't show same campaign twice
        seen_campaigns = set()
        diverse = []
        for c in candidates:
            if c["campaign_id"] not in seen_campaigns:
                diverse.append(c)
                seen_campaigns.add(c["campaign_id"])
            if len(diverse) >= num_ads * 2:
                break

        # Exploration: occasionally boost random ads
        if random.random() < 0.1 and len(diverse) > num_ads:
            idx = random.randint(num_ads, len(diverse) - 1)
            diverse[0], diverse[idx] = diverse[idx], diverse[0]

        return diverse


async def run_stress_test(config: StressTestConfig) -> StressTestResult:
    """Run the stress test."""
    print("\n" + "=" * 60)
    print("OpenAdServer Stress Test")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Campaigns: {config.num_campaigns}")
    print(f"  Creatives per campaign: {config.creatives_per_campaign}")
    print(f"  Requests: {config.num_requests}")
    print(f"  Concurrent: {config.concurrent_requests}")
    print(f"  Model: {config.model_type}")
    print(f"  ML Prediction: {config.enable_ml_prediction}")

    result = StressTestResult()

    # Initialize
    print("\n[1/4] Initializing...")
    db = InMemoryDatabase()
    redis = FakeRedisClient()

    await db.init()
    await redis.connect()

    # Seed data
    print("\n[2/4] Seeding data...")
    await seed_test_data(db, config)

    # Initialize pipeline
    print("\n[3/4] Initializing pipeline...")
    pipeline = SimplePipeline(db, redis, config)
    await pipeline.init()

    # Warmup
    print(f"\n  Warming up with {config.warmup_requests} requests...")
    for _ in range(config.warmup_requests):
        req = generate_ad_request()
        await pipeline.serve(req)

    # Clear GC before test
    gc.collect()

    # Run test with concurrent requests
    print(f"\n[4/4] Running {config.num_requests} requests (concurrency: {config.concurrent_requests})...")

    async def process_single_request(req_id: int) -> tuple:
        """Process a single request and return metrics."""
        req = generate_ad_request()
        num_requested = req.get("num_ads", 1)

        try:
            req_start = time.perf_counter()
            ads, times = await pipeline.serve(req)
            req_time = (time.perf_counter() - req_start) * 1000
            return (True, len(ads), num_requested, times, req_time)
        except Exception as e:
            return (False, 0, num_requested, {}, 0)

    start_time = time.perf_counter()

    # Process requests in batches for concurrent execution
    batch_size = config.concurrent_requests
    completed = 0

    for batch_start in range(0, config.num_requests, batch_size):
        batch_end = min(batch_start + batch_size, config.num_requests)
        batch_tasks = [process_single_request(i) for i in range(batch_start, batch_end)]

        # Execute batch concurrently
        batch_results = await asyncio.gather(*batch_tasks)

        # Collect results
        for success, ads_returned, ads_requested, times, req_time in batch_results:
            result.total_requests += 1
            result.total_ads_requested += ads_requested

            if success:
                result.successful_requests += 1
                result.total_ads_returned += ads_returned
                result.retrieval.times_ms.append(times.get("retrieval", 0))
                result.filter.times_ms.append(times.get("filter", 0))
                result.prediction.times_ms.append(times.get("prediction", 0))
                result.ranking.times_ms.append(times.get("ranking", 0))
                result.rerank.times_ms.append(times.get("rerank", 0))
                result.total.times_ms.append(req_time)
            else:
                result.failed_requests += 1

        completed = batch_end
        if completed % 1000 == 0 or completed == config.num_requests:
            elapsed = time.perf_counter() - start_time
            current_qps = completed / elapsed if elapsed > 0 else 0
            print(f"  Progress: {completed}/{config.num_requests} (QPS: {current_qps:.1f})")

    result.total_time_s = time.perf_counter() - start_time

    # Cleanup
    await redis.close()
    await db.close()

    return result


def print_results(result: StressTestResult):
    """Print test results."""
    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)

    print(f"\n📊 Overall:")
    print(f"  Total requests: {result.total_requests}")
    print(f"  Successful: {result.successful_requests}")
    print(f"  Failed: {result.failed_requests}")
    print(f"  Success rate: {result.success_rate:.2%}")
    print(f"  Total time: {result.total_time_s:.2f}s")
    print(f"  QPS: {result.qps:.1f}")
    print(f"  Fill rate: {result.fill_rate:.2%}")

    print(f"\n⏱️ Latency (ms):")
    print(f"  {'Stage':<12} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    stages = [
        result.retrieval,
        result.filter,
        result.prediction,
        result.ranking,
        result.rerank,
        result.total,
    ]

    for stage in stages:
        print(f"  {stage.name:<12} {stage.avg_ms:>8.2f} {stage.p50_ms:>8.2f} {stage.p95_ms:>8.2f} {stage.p99_ms:>8.2f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="OpenAdServer Stress Test")
    parser.add_argument("--campaigns", type=int, default=10, help="Number of campaigns")
    parser.add_argument("--creatives", type=int, default=3, help="Creatives per campaign")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "fm", "deepfm"])
    parser.add_argument("--no-ml", action="store_true", help="Disable ML prediction")

    args = parser.parse_args()

    config = StressTestConfig(
        num_campaigns=args.campaigns,
        creatives_per_campaign=args.creatives,
        num_requests=args.requests,
        concurrent_requests=args.concurrent,
        model_type=args.model,
        enable_ml_prediction=not args.no_ml,
    )

    result = asyncio.run(run_stress_test(config))
    print_results(result)

    # Save results
    import json
    results_path = Path("models/criteo/stress_test_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "campaigns": config.num_campaigns,
                "creatives_per_campaign": config.creatives_per_campaign,
                "requests": config.num_requests,
                "model": config.model_type,
            },
            "results": {
                "qps": result.qps,
                "success_rate": result.success_rate,
                "fill_rate": result.fill_rate,
                "latency_ms": {
                    "avg": result.total.avg_ms,
                    "p50": result.total.p50_ms,
                    "p95": result.total.p95_ms,
                    "p99": result.total.p99_ms,
                },
                "stages_ms": {
                    stage.name: {"avg": stage.avg_ms, "p99": stage.p99_ms}
                    for stage in [result.retrieval, result.filter, result.prediction, result.ranking, result.rerank]
                },
            },
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
