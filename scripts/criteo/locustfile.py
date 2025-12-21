#!/usr/bin/env python3
"""
Locust load testing for OpenAdServer.

Tests the complete ad serving flow:
1. Targeting/Retrieval
2. Filtering (Budget, Frequency, Quality)
3. CTR/CVR Prediction
4. Ranking/Bidding
5. Re-ranking (Diversity, Exploration)

Usage:
    # Start server first:
    python scripts/criteo/start_server.py

    # Run load test:
    locust -f scripts/criteo/locustfile.py --host http://localhost:8000

    # Or headless mode:
    locust -f scripts/criteo/locustfile.py --host http://localhost:8000 \
           --headless -u 100 -r 10 -t 60s
"""

import random
import string
import time
from typing import Any

from locust import HttpUser, between, task


class AdRequestUser(HttpUser):
    """Simulates users making ad requests."""

    # Wait 0.1-0.5 seconds between requests
    wait_time = between(0.1, 0.5)

    # Pre-defined data for realistic requests
    SLOT_IDS = [
        "banner_home_top",
        "banner_home_bottom",
        "native_feed",
        "interstitial_game",
        "video_preroll",
        "banner_detail",
        "native_article",
        "splash_screen",
    ]

    OS_OPTIONS = ["android", "ios"]
    OS_VERSIONS = {
        "android": ["10", "11", "12", "13", "14"],
        "ios": ["14.0", "15.0", "16.0", "17.0"],
    }

    DEVICE_BRANDS = {
        "android": ["Samsung", "Xiaomi", "Huawei", "OPPO", "vivo", "OnePlus", "Google"],
        "ios": ["Apple"],
    }

    DEVICE_MODELS = {
        "Samsung": ["Galaxy S21", "Galaxy S22", "Galaxy S23", "Galaxy A52"],
        "Xiaomi": ["Mi 11", "Mi 12", "Redmi Note 10", "POCO X3"],
        "Huawei": ["P40", "P50", "Mate 40", "Nova 9"],
        "OPPO": ["Reno6", "Find X3", "A96"],
        "vivo": ["X70", "V23", "Y21"],
        "OnePlus": ["9 Pro", "10 Pro", "Nord 2"],
        "Google": ["Pixel 6", "Pixel 7", "Pixel 7a"],
        "Apple": ["iPhone 12", "iPhone 13", "iPhone 14", "iPhone 15"],
    }

    COUNTRIES = ["CN", "US", "JP", "KR", "IN", "BR", "DE", "GB"]
    CITIES = {
        "CN": ["shanghai", "beijing", "guangzhou", "shenzhen", "hangzhou"],
        "US": ["new york", "los angeles", "chicago", "houston", "phoenix"],
        "JP": ["tokyo", "osaka", "kyoto", "yokohama"],
        "KR": ["seoul", "busan", "incheon"],
        "IN": ["mumbai", "delhi", "bangalore", "hyderabad"],
        "BR": ["sao paulo", "rio de janeiro", "brasilia"],
        "DE": ["berlin", "munich", "frankfurt", "hamburg"],
        "GB": ["london", "manchester", "birmingham"],
    }

    NETWORKS = ["wifi", "4g", "5g", "3g"]
    CARRIERS = ["China Mobile", "China Unicom", "China Telecom", "AT&T", "Verizon"]

    INTERESTS = [
        "gaming", "sports", "music", "movies", "travel",
        "food", "fashion", "technology", "finance", "education",
        "health", "fitness", "shopping", "news", "social",
    ]

    APP_CATEGORIES = [
        "game", "social", "entertainment", "tools", "lifestyle",
        "education", "finance", "health", "shopping", "news",
    ]

    def on_start(self):
        """Initialize user session."""
        # Generate a persistent user ID for this session
        self.user_id = self._generate_user_id()
        self.session_start = time.time()
        self.request_count = 0

    def _generate_user_id(self) -> str:
        """Generate a realistic user ID."""
        return "user_" + "".join(random.choices(string.hexdigits.lower(), k=16))

    def _generate_device_info(self) -> dict[str, Any]:
        """Generate realistic device information."""
        os = random.choice(self.OS_OPTIONS)
        brand = random.choice(self.DEVICE_BRANDS[os])
        model = random.choice(self.DEVICE_MODELS[brand])

        return {
            "os": os,
            "os_version": random.choice(self.OS_VERSIONS[os]),
            "model": model,
            "brand": brand,
            "screen_width": random.choice([720, 1080, 1440, 2160]),
            "screen_height": random.choice([1280, 1920, 2560, 3840]),
            "language": random.choice(["zh-CN", "en-US", "ja-JP", "ko-KR"]),
        }

    def _generate_geo_info(self) -> dict[str, Any]:
        """Generate realistic geo information."""
        country = random.choice(self.COUNTRIES)
        city = random.choice(self.CITIES.get(country, ["unknown"]))

        return {
            "country": country,
            "city": city,
            "latitude": random.uniform(-90, 90),
            "longitude": random.uniform(-180, 180),
        }

    def _generate_context_info(self) -> dict[str, Any]:
        """Generate context information."""
        return {
            "app_id": f"com.example.app{random.randint(1, 100)}",
            "app_name": f"TestApp{random.randint(1, 100)}",
            "app_version": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            "network": random.choice(self.NETWORKS),
            "carrier": random.choice(self.CARRIERS),
        }

    def _generate_user_features(self) -> dict[str, Any]:
        """Generate user features for ML prediction."""
        return {
            "age": random.randint(18, 65),
            "gender": random.choice(["male", "female", "unknown"]),
            "interests": random.sample(self.INTERESTS, k=random.randint(2, 5)),
            "app_categories": random.sample(self.APP_CATEGORIES, k=random.randint(1, 3)),
        }

    def _build_ad_request(self, num_ads: int = 1) -> dict[str, Any]:
        """Build a complete ad request."""
        return {
            "slot_id": random.choice(self.SLOT_IDS),
            "user_id": self.user_id,
            "device": self._generate_device_info(),
            "geo": self._generate_geo_info(),
            "context": self._generate_context_info(),
            "user_features": self._generate_user_features(),
            "num_ads": num_ads,
        }

    @task(10)
    def request_single_ad(self):
        """Request a single ad (most common)."""
        request_data = self._build_ad_request(num_ads=1)

        with self.client.post(
            "/api/v1/ad/request",
            json=request_data,
            name="/api/v1/ad/request (single)",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("count", 0) >= 0:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")

        self.request_count += 1

    @task(3)
    def request_multiple_ads(self):
        """Request multiple ads (for feed/carousel)."""
        num_ads = random.choice([3, 5])
        request_data = self._build_ad_request(num_ads=num_ads)

        with self.client.post(
            "/api/v1/ad/request",
            json=request_data,
            name=f"/api/v1/ad/request (multi:{num_ads})",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("count", 0) >= 0:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")

        self.request_count += 1

    @task(1)
    def health_check(self):
        """Periodic health check."""
        self.client.get("/health", name="/health")

    @task(1)
    def ready_check(self):
        """Readiness check."""
        self.client.get("/ready", name="/ready")


class EventTrackingUser(HttpUser):
    """Simulates event tracking (impressions, clicks)."""

    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Initialize tracking user."""
        self.user_id = "user_" + "".join(random.choices(string.hexdigits.lower(), k=16))

    def _generate_request_id(self) -> str:
        """Generate a request ID."""
        return "req_" + "".join(random.choices(string.hexdigits.lower(), k=12))

    @task(5)
    def track_impression(self):
        """Track ad impression event."""
        params = {
            "type": "impression",
            "req": self._generate_request_id(),
            "ad": random.randint(1, 10),
        }

        self.client.get(
            "/api/v1/event/track",
            params=params,
            name="/api/v1/event/track (impression)",
        )

    @task(1)
    def track_click(self):
        """Track click event (less frequent than impressions)."""
        params = {
            "type": "click",
            "req": self._generate_request_id(),
            "ad": random.randint(1, 10),
        }

        self.client.get(
            "/api/v1/event/track",
            params=params,
            name="/api/v1/event/track (click)",
        )


class MixedWorkloadUser(HttpUser):
    """Mixed workload simulating real traffic patterns."""

    wait_time = between(0.1, 1.0)

    # Traffic distribution weights
    tasks = {
        "ad_request": 70,    # 70% ad requests
        "impression": 20,    # 20% impression tracking
        "click": 5,          # 5% click tracking
        "health": 5,         # 5% health checks
    }

    SLOT_IDS = [
        "banner_home_top", "banner_home_bottom", "native_feed",
        "interstitial_game", "video_preroll", "banner_detail",
    ]

    def on_start(self):
        self.user_id = "user_" + "".join(random.choices(string.hexdigits.lower(), k=16))

    @task(70)
    def ad_request(self):
        """Primary ad request task."""
        request_data = {
            "slot_id": random.choice(self.SLOT_IDS),
            "user_id": self.user_id,
            "device": {
                "os": random.choice(["android", "ios"]),
                "os_version": "13.0",
                "model": "TestDevice",
                "brand": "TestBrand",
            },
            "geo": {
                "country": random.choice(["CN", "US", "JP"]),
                "city": "shanghai",
            },
            "num_ads": random.choice([1, 1, 1, 3]),  # Weighted towards single ads
        }

        with self.client.post(
            "/api/v1/ad/request",
            json=request_data,
            name="/api/v1/ad/request",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(20)
    def track_impression(self):
        """Impression tracking."""
        params = {
            "type": "impression",
            "req": "req_" + "".join(random.choices(string.hexdigits.lower(), k=8)),
            "ad": random.randint(1, 5),
        }
        self.client.get("/api/v1/event/track", params=params, name="/event/impression")

    @task(5)
    def track_click(self):
        """Click tracking."""
        params = {
            "type": "click",
            "req": "req_" + "".join(random.choices(string.hexdigits.lower(), k=8)),
            "ad": random.randint(1, 5),
        }
        self.client.get("/api/v1/event/track", params=params, name="/event/click")

    @task(5)
    def health_check(self):
        """Health check."""
        self.client.get("/health", name="/health")
