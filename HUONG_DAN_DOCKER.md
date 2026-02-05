# Hướng Dẫn Chạy OpenAdServer với Docker

## ⚡ Quick Start (Bắt Đầu Nhanh)

```bash
# 1. Clone repository
git clone https://github.com/pysean/openadserver.git
cd openadserver

# 2. Khởi động tất cả services
docker compose up -d

# 3. Kiểm tra health
curl http://localhost:8000/health

# 4. Test ad request
curl -X POST http://localhost:8000/api/v1/ad/request \
  -H "Content-Type: application/json" \
  -d '{"slot_id": "banner_home", "user_id": "user_123", "num_ads": 1}'
```

**Các URL quan trọng:**
- Ad Server API: http://localhost:8000
- API Docs: http://localhost:8000/docs (khi debug=true)
- Grafana: http://localhost:3000 (admin/admin) - cần `--profile monitoring`
- Prometheus: http://localhost:9090 - cần `--profile monitoring`

---

## 📋 Tổng Quan Dự Án

**OpenAdServer** (hay **LiteAds**) là một nền tảng quảng cáo mã nguồn mở, tự host với khả năng dự đoán CTR (Click-Through Rate) bằng Machine Learning. Đây là giải pháp hoàn chỉnh cho các doanh nghiệp vừa và nhỏ muốn tự quản lý hệ thống quảng cáo của mình.

### 🎯 Đối Tượng Sử Dụng

- **SMBs**: Xây dựng mạng quảng cáo riêng
- **Công ty Game**: Monetize traffic trong ứng dụng
- **App Developers**: Chạy house ads hoặc direct deals
- **E-commerce**: Sponsored listings
- **Researchers**: Nghiên cứu computational advertising
- **Students**: Học về ad-tech systems

---

## ✨ Các Tính Năng Chính

### 🚀 Ad Serving (Phục Vụ Quảng Cáo)

1. **High-Performance API**
   - Latency P99 < 10ms với FastAPI
   - Hỗ trợ async/await cho throughput cao
   - Auto-scaling với Docker Compose

2. **Multiple Ad Formats**
   - Banner ads
   - Native ads
   - Video ads (roadmap)
   - Interstitial ads

3. **Smart Targeting**
   - Geo targeting (quốc gia, thành phố)
   - Device targeting (OS, version, model)
   - Demographics (age, gender)
   - Interests & behaviors
   - Custom targeting rules

4. **Frequency Capping**
   - Daily cap (giới hạn số lần hiển thị/ngày)
   - Hourly cap (giới hạn số lần hiển thị/giờ)
   - Per-user tracking với Redis

5. **Budget Pacing**
   - Daily budget management
   - Total budget tracking
   - Smooth delivery trong ngày

### 🤖 Machine Learning

1. **CTR Prediction Models**
   - **Logistic Regression (LR)**: Nhanh nhất, AUC tốt nhất (0.7577)
   - **Factorization Machine (FM)**: Capture feature interactions
   - **DeepFM**: Deep learning + FM kết hợp

2. **Real-time Inference**
   - Prediction latency < 5ms
   - Batch prediction cho hiệu suất cao
   - Model hot-swap (cập nhật model không downtime)

3. **Feature Engineering**
   - Sparse features: 26 categorical features
   - Dense features: 13 numerical features
   - Numba JIT acceleration
   - Automatic feature hashing & encoding

### 💰 Monetization

1. **eCPM Ranking**
   - Tự động maximize revenue
   - Công thức: eCPM = bid × pCTR × 1000
   - Hỗ trợ multiple bid types

2. **Bid Types**
   - **CPM** (Cost Per Mille): Trả theo 1000 impressions
   - **CPC** (Cost Per Click): Trả theo click
   - **CPA** (Cost Per Action): Trả theo conversion
   - **oCPM** (Optimized CPM): Tối ưu tự động

3. **Real-time Bidding**
   - OpenRTB compatible (roadmap)
   - Auction-based selection

### 📊 Analytics & Monitoring

1. **Event Tracking**
   - Impressions (hiển thị)
   - Clicks (nhấp chuột)
   - Conversions (chuyển đổi)
   - Real-time logging

2. **Prometheus Metrics**
   - Request rate (QPS)
   - Latency (P50, P95, P99)
   - Error rate
   - Business metrics (impressions, clicks, revenue)

3. **Grafana Dashboards**
   - Real-time monitoring
   - Performance metrics
   - Business analytics

---

## 🏗️ Kiến Trúc Hệ Thống

### Pipeline Xử Lý Ad Request

```
┌─────────────────────────────────────────────────────────────┐
│                    Ad Request Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   📱 Client Request                                         │
│      │                                                      │
│      ▼                                                      │
│   ┌──────────┐    ┌───────────┐    ┌──────────┐          │
│   │ FastAPI  │───▶│ Retrieval  │───▶│ Ranking  │          │
│   │  Router  │    │(Targeting)│    │ (eCPM)   │          │
│   └──────────┘    └───────────┘    └──────────┘          │
│        │               │                │                  │
│        ▼               ▼                ▼                  │
│   ┌──────────┐    ┌───────────┐    ┌──────────┐          │
│   │PostgreSQL│    │   Redis   │    │ PyTorch  │          │
│   │(Campaigns)│   │  (Cache)   │    │ (Models) │          │
│   └──────────┘    └───────────┘    └──────────┘          │
│                                                             │
│   Pipeline: Retrieve → Filter → Predict → Rank → Return    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Các Bước Xử Lý

1. **Retrieval (Thu thập ứng viên)**
   - Query PostgreSQL để lấy campaigns phù hợp
   - Áp dụng targeting rules (geo, device, demographics)
   - Trả về ~100 candidates

2. **Filtering (Lọc)**
   - **Budget Filter**: Kiểm tra budget còn lại
   - **Frequency Filter**: Kiểm tra frequency cap (Redis)
   - **Quality Filter**: Loại bỏ ads chất lượng thấp

3. **Prediction (Dự đoán)**
   - Predict CTR (pCTR) bằng ML model
   - Predict CVR (pCVR) nếu có
   - Fallback về statistical predictor nếu không có model

4. **Ranking (Xếp hạng)**
   - Tính eCPM = bid × pCTR × 1000
   - Sắp xếp theo eCPM giảm dần
   - Loại bỏ ads có eCPM quá thấp

5. **Re-ranking (Xếp hạng lại)**
   - **Diversity Reranker**: Đảm bảo đa dạng campaigns
   - **Exploration Reranker**: Thử nghiệm ads mới (epsilon-greedy)
   - Chọn top N ads cuối cùng

### Cấu Trúc Thư Mục

```
openadserver/
├── liteads/
│   ├── ad_server/          # FastAPI application
│   │   ├── routers/        # API endpoints (ad, event, health)
│   │   ├── services/       # Business logic
│   │   └── middleware/     # Logging, metrics, auth
│   ├── rec_engine/         # Recommendation engine
│   │   ├── retrieval/      # Candidate retrieval & targeting
│   │   ├── ranking/         # eCPM bidding & ranking
│   │   ├── filter/         # Budget, frequency, quality filters
│   │   └── reranking/      # Diversity & exploration
│   ├── ml_engine/          # Machine learning
│   │   ├── models/         # DeepFM, LR, FM implementations
│   │   ├── features/       # Feature engineering pipeline
│   │   └── serving/        # Online prediction server
│   ├── common/             # Shared utilities
│   │   ├── config.py       # Configuration management
│   │   ├── database.py     # PostgreSQL connection
│   │   ├── cache.py        # Redis client
│   │   └── logger.py       # Structured logging
│   └── schemas/            # Pydantic models
├── configs/                # YAML configurations
├── deployment/             # Docker, Nginx, Prometheus, Grafana
├── scripts/                # Utility scripts
└── tests/                  # Test suite
```

---

## 🐳 Hướng Dẫn Chạy Docker Local

### Yêu Cầu Hệ Thống

- **Docker**: >= 20.10
- **Docker Compose**: >= 2.0
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Disk**: Tối thiểu 5GB trống
- **OS**: Linux, macOS, hoặc Windows với WSL2

### Bước 1: Clone Repository

```bash
git clone https://github.com/pysean/openadserver.git
cd openadserver
```

### Bước 2: Kiểm Tra Docker

```bash
# Kiểm tra Docker đã cài đặt
docker --version
docker compose version

# Kiểm tra Docker đang chạy
docker ps
```

### Bước 3: Khởi Động Services

#### Option A: Chạy Tất Cả Services (Khuyến Nghị)

```bash
# Khởi động tất cả services: PostgreSQL, Redis, Ad Server
docker compose up -d

# Xem logs
docker compose logs -f ad-server

# Kiểm tra trạng thái
docker compose ps
```

#### Option B: Chạy Chỉ Core Services

```bash
# Chỉ chạy PostgreSQL và Redis
docker compose up -d postgres redis

# Chạy Ad Server local (cần Python 3.11+)
pip install -e ".[dev]"
LITEADS_ENV=dev python -m liteads.ad_server.main
```

#### Option C: Chạy Với Monitoring (Prometheus + Grafana)

```bash
# Khởi động với monitoring
docker compose --profile monitoring up -d

# Truy cập:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Bước 4: Khởi Tạo Database

Database sẽ tự động được khởi tạo khi PostgreSQL container start lần đầu (từ file `scripts/init_db.sql`). Nếu cần khởi tạo lại:

```bash
# Khởi tạo database
make db-init

# Hoặc thủ công
docker compose exec postgres psql -U liteads -d liteads -f /docker-entrypoint-initdb.d/init.sql
```

### Bước 5: Tạo Dữ Liệu Mẫu (Optional)

```bash
# Tạo mock data
python scripts/init_test_data.py

# Hoặc tạo nhiều data hơn
python scripts/generate_mock_data.py --advertisers 10 --campaigns 5 --creatives 3
```

### Bước 6: Kiểm Tra Health

```bash
# Health check
curl http://localhost:8000/health

# Kết quả mong đợi:
# {"status":"healthy","version":"0.1.0"}

# Hoặc dùng Makefile
make health
```

### Bước 7: Test Ad Request

```bash
# Gửi ad request
curl -X POST http://localhost:8000/api/v1/ad/request \
  -H "Content-Type: application/json" \
  -d '{
    "slot_id": "banner_home",
    "user_id": "user_12345",
    "device": {"os": "ios", "os_version": "17.0"},
    "geo": {"country": "US", "city": "new_york"},
    "num_ads": 3
  }'
```

**Response mẫu:**
```json
{
  "request_id": "req_a1b2c3d4",
  "ads": [
    {
      "ad_id": "ad_1001_5001",
      "campaign_id": 1001,
      "creative": {
        "title": "Summer Sale - 50% Off!",
        "description": "Limited time offer",
        "image_url": "https://cdn.example.com/ads/summer-sale.jpg",
        "landing_url": "https://shop.example.com/sale"
      },
      "tracking": {
        "impression_url": "http://localhost:8000/api/v1/event/track?type=impression&req=req_a1b2c3d4&ad=1001",
        "click_url": "http://localhost:8000/api/v1/event/track?type=click&req=req_a1b2c3d4&ad=1001"
      },
      "metadata": {
        "ecpm": 35.50,
        "pctr": 0.0355
      }
    }
  ],
  "count": 1
}
```

---

## 🔧 Cấu Hình

### Environment Variables

Các biến môi trường có thể được set trong `docker-compose.yml` hoặc file `.env`:

```bash
# Database
LITEADS_DATABASE__HOST=postgres
LITEADS_DATABASE__PORT=5432
LITEADS_DATABASE__NAME=liteads
LITEADS_DATABASE__USER=liteads
LITEADS_DATABASE__PASSWORD=liteads_password

# Redis
LITEADS_REDIS__HOST=redis
LITEADS_REDIS__PORT=6379

# Environment
LITEADS_ENV=prod  # hoặc dev
```

### Configuration Files

Cấu hình trong `configs/`:

- `base.yaml`: Cấu hình cơ bản
- `dev.yaml`: Development settings
- `prod.yaml`: Production settings

Ví dụ `configs/prod.yaml`:
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

ad_serving:
  enable_ml_prediction: true
  default_num_ads: 1
  max_num_ads: 10
  timeout_ms: 50

ml:
  model_dir: "./models"
  ctr_model: "deepfm_v1"
```

---

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Ad Request
```bash
POST /api/v1/ad/request
Content-Type: application/json

{
  "slot_id": "banner_home",
  "user_id": "user_123",
  "device": {"os": "ios", "os_version": "17.0"},
  "geo": {"country": "US", "city": "new_york"},
  "num_ads": 3
}
```

### Event Tracking
```bash
GET /api/v1/event/track?type=impression&req=req_123&ad=1001
GET /api/v1/event/track?type=click&req=req_123&ad=1001
GET /api/v1/event/track?type=conversion&req=req_123&ad=1001
```

### Metrics (Prometheus)
```bash
GET /metrics
```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs (chỉ khi `debug=true`)
- **ReDoc**: http://localhost:8000/redoc (chỉ khi `debug=true`)

---

## 🛠️ Các Lệnh Hữu Ích

### Docker Commands

```bash
# Khởi động services
make docker-up
# hoặc
docker compose up -d

# Dừng services
make docker-down
# hoặc
docker compose down

# Xem logs
make docker-logs
# hoặc
docker compose logs -f ad-server

# Rebuild images
make docker-build
# hoặc
docker compose build --no-cache

# Restart service
make docker-restart
# hoặc
docker compose restart ad-server

# Scale ad-server
docker compose up -d --scale ad-server=3
```

### Database Commands

```bash
# Khởi tạo database
make db-init

# Kết nối PostgreSQL shell
make db-shell
# hoặc
docker compose exec postgres psql -U liteads -d liteads

# Tạo mock data
make db-mock
```

### Redis Commands

```bash
# Kết nối Redis CLI
make redis-cli
# hoặc
docker compose exec redis redis-cli

# Xóa tất cả cache
make redis-flush
# hoặc
docker compose exec redis redis-cli FLUSHALL
```

### Development Commands

```bash
# Cài đặt dependencies
make install

# Chạy development server
make dev

# Chạy tests
make test

# Linting
make lint

# Format code
make format

# Clean build artifacts
make clean
```

---

## 📊 Monitoring & Observability

### Prometheus

Truy cập: http://localhost:9090

**Metrics quan trọng:**
- `http_requests_total`: Tổng số requests
- `http_request_duration_seconds`: Latency
- `ad_requests_total`: Tổng số ad requests
- `ad_impressions_total`: Tổng số impressions
- `ad_clicks_total`: Tổng số clicks

### Grafana

Truy cập: http://localhost:3000
- **Username**: `admin`
- **Password**: `admin` (đổi sau lần đăng nhập đầu)

**Dashboards có sẵn:**
- LiteAds Performance Dashboard
- Business Metrics Dashboard

### Logs

```bash
# Xem logs real-time
docker compose logs -f ad-server

# Xem logs của tất cả services
docker compose logs -f

# Xem logs của service cụ thể
docker compose logs -f postgres
docker compose logs -f redis
```

---

## 🧪 Testing

### Unit Tests

```bash
# Chạy tất cả tests
make test

# Chạy tests nhanh (dừng khi fail)
make test-fast

# Chạy tests với coverage
pytest tests/ -v --cov=liteads --cov-report=html
```

### Integration Tests

```bash
# E2E test
python scripts/test_full_flow.py

# Stress test
python scripts/criteo/stress_test.py --campaigns 200 --requests 10000
```

### Benchmark

```bash
# Benchmark với wrk (cần cài wrk)
make benchmark

# Hoặc dùng Locust
cd scripts/criteo
locust -f locustfile.py
```

---

## 🐛 Troubleshooting

### Lỗi: Port đã được sử dụng

```bash
# Kiểm tra port đang được dùng
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Đổi port trong docker-compose.yml
ports:
  - "8001:8000"  # Thay vì 8000:8000
```

### Lỗi: Database connection failed

```bash
# Kiểm tra PostgreSQL đã start
docker compose ps postgres

# Kiểm tra logs
docker compose logs postgres

# Restart PostgreSQL
docker compose restart postgres

# Kiểm tra kết nối
docker compose exec postgres psql -U liteads -d liteads -c "SELECT 1;"
```

### Lỗi: Redis connection failed

```bash
# Kiểm tra Redis đã start
docker compose ps redis

# Kiểm tra logs
docker compose logs redis

# Test Redis
docker compose exec redis redis-cli ping
# Kết quả: PONG
```

### Lỗi: Container không start

```bash
# Xem logs chi tiết
docker compose logs ad-server

# Rebuild image
docker compose build --no-cache ad-server

# Xóa volumes và khởi động lại
docker compose down -v
docker compose up -d
```

### Lỗi: Out of memory

```bash
# Giảm số workers trong configs/prod.yaml
server:
  workers: 2  # Thay vì 4

# Hoặc giảm memory limit cho containers
# Thêm vào docker-compose.yml:
services:
  ad-server:
    deploy:
      resources:
        limits:
          memory: 2G
```

---

## 📈 Performance Tuning

### Tối Ưu Database

```bash
# Tăng connection pool
# Trong configs/prod.yaml:
database:
  pool_size: 20
  max_overflow: 40
```

### Tối Ưu Redis

```bash
# Tăng memory limit
# Trong docker-compose.yml:
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Tối Ưu Ad Server

```bash
# Tăng số workers
# Trong configs/prod.yaml:
server:
  workers: 8  # Tùy theo CPU cores

# Hoặc scale horizontal
docker compose up -d --scale ad-server=3
```

---

## 🔐 Security

### Production Checklist

1. **Đổi passwords mặc định**
   ```bash
   # Trong docker-compose.yml
   POSTGRES_PASSWORD: your_secure_password
   GF_SECURITY_ADMIN_PASSWORD: your_secure_password
   ```

2. **Sử dụng secrets**
   ```bash
   # Tạo file .env
   DB_PASSWORD=your_secure_password
   REDIS_PASSWORD=your_secure_password
   
   # Trong docker-compose.yml
   environment:
     - LITEADS_DATABASE__PASSWORD=${DB_PASSWORD}
   ```

3. **Tắt debug mode**
   ```yaml
   # configs/prod.yaml
   app:
     debug: false
   ```

4. **CORS configuration**
   ```python
   # Chỉ cho phép domains cụ thể
   allow_origins=["https://yourdomain.com"]
   ```

5. **Network isolation**
   ```yaml
   # Chỉ expose ports cần thiết
   # Không expose PostgreSQL/Redis ra ngoài
   ```

---

## 📚 Tài Liệu Tham Khảo

- **README.md**: Tài liệu chính của project
- **API Docs**: http://localhost:8000/docs (khi debug=true)
- **Grafana Dashboards**: http://localhost:3000
- **Prometheus**: http://localhost:9090

---

## 🆘 Hỗ Trợ

Nếu gặp vấn đề:

1. Kiểm tra logs: `docker compose logs -f`
2. Kiểm tra health: `curl http://localhost:8000/health`
3. Xem troubleshooting section ở trên
4. Tạo issue trên GitHub: https://github.com/pysean/openadserver/issues

---

## 🎉 Kết Luận

Bạn đã hoàn tất setup OpenAdServer với Docker! Hệ thống bao gồm:

- ✅ Ad Server (FastAPI)
- ✅ PostgreSQL Database
- ✅ Redis Cache
- ✅ Prometheus Monitoring (optional)
- ✅ Grafana Dashboards (optional)

**Next Steps:**
1. Tạo campaigns và creatives qua API hoặc database
2. Train ML models với dữ liệu của bạn
3. Tích hợp vào ứng dụng của bạn
4. Monitor performance qua Grafana

---

## 📊 Tóm Tắt Thông Tin Quan Trọng

### Services & Ports

| Service | Port | URL | Mô Tả |
|---------|------|-----|-------|
| Ad Server | 8000 | http://localhost:8000 | FastAPI application |
| PostgreSQL | 5432 | - | Database (internal) |
| Redis | 6379 | - | Cache (internal) |
| Nginx | 80 | http://localhost | Load balancer (production profile) |
| Prometheus | 9090 | http://localhost:9090 | Metrics (monitoring profile) |
| Grafana | 3000 | http://localhost:3000 | Dashboards (monitoring profile) |

### Environment Variables

| Variable | Mặc Định | Mô Tả |
|----------|----------|-------|
| `LITEADS_ENV` | `dev` | Environment (dev/prod) |
| `LITEADS_DATABASE__HOST` | `postgres` | PostgreSQL host |
| `LITEADS_DATABASE__PORT` | `5432` | PostgreSQL port |
| `LITEADS_DATABASE__NAME` | `liteads` | Database name |
| `LITEADS_DATABASE__USER` | `liteads` | Database user |
| `LITEADS_DATABASE__PASSWORD` | `liteads_password` | Database password |
| `LITEADS_REDIS__HOST` | `redis` | Redis host |
| `LITEADS_REDIS__PORT` | `6379` | Redis port |

### API Endpoints

| Endpoint | Method | Mô Tả |
|----------|--------|-------|
| `/health` | GET | Health check |
| `/api/v1/ad/request` | POST | Request ads |
| `/api/v1/event/track` | GET | Track events |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI (debug only) |
| `/redoc` | GET | ReDoc (debug only) |

### Docker Compose Profiles

| Profile | Services | Command |
|---------|-----------|---------|
| Default | ad-server, postgres, redis | `docker compose up -d` |
| Monitoring | + prometheus, grafana | `docker compose --profile monitoring up -d` |
| Production | + nginx | `docker compose --profile production up -d` |
| Full | All services | `docker compose --profile monitoring --profile production up -d` |

### Makefile Commands

| Command | Mô Tả |
|---------|-------|
| `make docker-up` | Khởi động services |
| `make docker-down` | Dừng services |
| `make docker-logs` | Xem logs |
| `make db-init` | Khởi tạo database |
| `make db-mock` | Tạo mock data |
| `make test` | Chạy tests |
| `make health` | Health check |

### Performance Benchmarks

| Model | QPS | Avg Latency | P99 | AUC |
|-------|-----|-------------|-----|-----|
| LR | 189.7 | 5.24ms | 10.02ms | 0.7577 |
| FM | 166.1 | 5.99ms | 11.54ms | 0.7472 |
| DeepFM | 151.2 | 6.58ms | 14.13ms | 0.7178 |

> **Khuyến nghị**: Sử dụng LR model cho production (nhanh nhất, AUC tốt nhất)

---

Chúc bạn thành công! 🚀
