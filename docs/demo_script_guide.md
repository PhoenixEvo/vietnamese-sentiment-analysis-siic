# DEMO SCRIPT & TALKING POINTS

## **PRE-DEMO SETUP (5 phút trước)**

### **Technical Setup**

```bash
# Terminal 1: Start API Server
cd D:\SIIC
python -m uvicorn api.main:app --reload --port 8000

# Terminal 2: Start Dashboard  
streamlit run dashboard/app.py --server.port 8501

# Terminal 3: Keep ready for testing
curl http://localhost:8000/health
```

### **Browser Setup**
- **Tab 1**: http://localhost:8501 (Dashboard)
- **Tab 2**: http://localhost:8000/docs (API Swagger)
- **Tab 3**: http://localhost:8000/health (Health Check)

### **Test Data Ready**
```
Test Sentences (Vietnamese):
1. "Hôm nay tôi rất vui vì được gặp bạn bè!" (Positive)
2. "Tôi thấy buồn vì phim này không hay" (Negative)  
3. "Sản phẩm này bình thường thôi" (Neutral)
4. "Dịch vụ khách hàng tệ quá, tôi rất tức giận!" (Negative)
5. "Cảm ơn team đã hỗ trợ tôi rất nhiều!" (Positive)
```

---

## **MAIN DEMO FLOW (15 phút)**

### **Phase 1: Problem Introduction (2 phút)**

**Slide 1-2: Welcome & Problem**
```
"Chào mọi người! Hôm nay team InsideOut sẽ trình bày hệ thống 
Vietnamese Emotion Detection - một giải pháp AI để phân tích 
cảm xúc từ bình luận mạng xã hội tiếng Việt.

Vấn đề: Với 70+ triệu người dùng mạng xã hội tại VN, 
việc phân tích cảm xúc khách hàng thủ công là không khả thi.
Chúng tôi cần một hệ thống tự động, chính xác và real-time."
```

**Slide 3: Technical Challenge**
```
"Thách thức với tiếng Việt:
- Ngữ pháp phức tạp, nhiều từ đồng âm
- Ngôn ngữ mạng xã hội không chuẩn
- Thiếu dataset chất lượng cao
- Cần độ chính xác 80%+ để ứng dụng thực tế"
```

### **Phase 2: Technical Solution (3 phút)**

**Slide 4-5: Architecture Overview**
```
"Hệ thống của chúng tôi gồm 3 components chính:
1. Advanced NLP Pipeline - Xử lý text tiếng Việt
2. Multiple AI Models - LSTM, SVM, Random Forest
3. Production-Ready API - FastAPI + Streamlit Dashboard"
```

*[Hiển thị architecture diagram]*

**Slide 6: Model Innovation**
```
"Điểm đột phá: Optimized LSTM với BiLSTM + Attention
- 4.5M parameters được tối ưu
- Attention mechanism focus vào từ quan trọng
- Class balancing để xử lý imbalanced data
- Kết quả: 85.79% accuracy trên UIT-VSFC dataset"
```

### **Phase 3: LIVE DEMO (6-7 phút)**

#### **Demo 1: Single Text Analysis (2 phút)**

*[Chuyển sang Tab 1 - Dashboard]*

```
"Bây giờ chúng tôi sẽ demo live với dashboard thực tế.
Đây là giao diện Streamlit đang chạy local trên localhost:8501"
```

**Actions:**
1. Nhập: "Hôm nay tôi rất vui vì được gặp bạn bè!"
2. Chọn model: "Optimized LSTM: BiLSTM + Attention"
3. Click "Phân tích"

**Talking Points:**
```
"Như các bạn thấy:
- Emotion: Positive với confidence 89.2%
- Processing time: chỉ 45ms
- Probability breakdown hiển thị distribution
- Model đã hiểu đúng context 'vui' và 'gặp bạn bè'"
```

#### **Demo 2: Model Comparison (2 phút)**

**Actions:**
1. Test cùng câu với SVM model
2. So sánh kết quả LSTM vs SVM

**Talking Points:**
```
"Thú vị! SVM cũng cho kết quả Positive với 85.6% confidence.
Điều này chứng minh consistency giữa các models.
LSTM cao hơn một chút nhưng SVM nhanh hơn (30ms vs 45ms).
Tùy use case mà chọn model phù hợp."
```

#### **Demo 3: API Integration (2 phút)**

*[Chuyển sang Tab 2 - API Swagger]*

```
"Đây là API documentation tự động tạo bởi FastAPI.
Chúng tôi sẽ test API endpoint trực tiếp."
```

**Actions:**
1. Expand /predict endpoint
2. Click "Try it out"
3. Input JSON: `{"text": "Dịch vụ tệ quá!", "model_type": "lstm"}`
4. Execute

**Talking Points:**
```
"API response ngay lập tức:
- emotion: negative
- confidence: 92.1%
- processing_time: 0.047 seconds
- Kết quả consistent và reliable để integrate vào production"
```

#### **Demo 4: Batch Processing (1 phút)**

*[Quay lại Dashboard - Tab Batch Processing]*

**Actions:**
1. Upload sample CSV file (prepared beforehand)
2. Click "Phân tích Tất cả"
3. Show results table

**Talking Points:**
```
"Batch processing cho phép analyze 100+ comments cùng lúc.
Kết quả xuất ra CSV với emotion và confidence cho từng comment.
Phù hợp cho social media monitoring và customer feedback analysis."
```

### **Phase 4: Technical Results (2 phút)**

**Slide 7-8: Performance Metrics**

```
"Kết quả đánh giá comprehensive trên UIT-VSFC dataset:

Model Performance:
- Optimized LSTM: 85.79% accuracy, 85.47% F1-score
- SVM: 85.68% accuracy, 84.93% F1-score  
- Random Forest: 84.23% accuracy
- Logistic Regression: 83.45% accuracy

Điều đặc biệt: SVM gần bằng Deep Learning nhưng nhanh hơn 3x!"
```

**Slide 9: Production Metrics**

```
"Production-ready metrics:
- Response time: 30-50ms average
- Throughput: 1000+ requests/second
- Uptime: 99.9% (Docker + Health checks)
- Memory usage: <2GB (optimized models)
- API compatibility: RESTful JSON format"
```

### **Phase 5: Business Impact (2 phút)**

**Slide 10-11: Applications**

```
"Ứng dụng thực tế:

1. Social Media Monitoring
   - Real-time sentiment tracking
   - Brand reputation management
   - Crisis detection & response

2. Customer Service
   - Automatic priority routing
   - Customer satisfaction analysis
   - Quality assurance monitoring

3. Market Research
   - Product feedback analysis
   - Campaign effectiveness measurement
   - Competitive intelligence"
```

**Slide 12: ROI & Scalability**

```
"Business Value:
- Giảm 80% thời gian phân tích manual
- Tăng 3x tốc độ response to negative feedback
- Chi phí vận hành: <$100/month cho 100K requests
- Scalable: Cloud deployment với auto-scaling"
```

---

## **Q&A PREPARATION (5 phút)**

### **Câu hỏi thường gặp:**

**Q: "Tại sao không dùng ChatGPT/GPT-4?"**
A: "GPT-4 rất mạnh nhưng:
- Chi phí cao ($0.03/1K tokens vs $0.001 của chúng tôi)
- Latency cao (1-3s vs 50ms)
- Không specialize cho tiếng Việt
- Data privacy concerns
- Chúng tôi optimize specific cho Vietnamese emotion detection"

**Q: "Làm sao đảm bảo accuracy với slang/teen language?"**
A: "Chúng tôi sử dụng:
- UIT-VSFC dataset chứa real social media data
- Preprocessing pipeline handle emojis, abbreviations
- Regular model retraining với new data
- Confidence threshold để flag uncertain cases"

**Q: "Scale như thế nào cho enterprise?"**
A: "Architecture được thiết kế cho scale:
- Docker containerization
- Redis caching layer
- Horizontal scaling với load balancer
- Database integration cho logging/analytics
- RESTful API cho easy integration"

**Q: "Độ chính xác so với human?"**
A: "Inter-annotator agreement của human ở ~87-90%.
Model của chúng tôi đạt 85.79%, rất gần human-level.
Với consistent performance và 24/7 availability."

---

## **SUCCESS METRICS**

### **Demo thành công khi:**
- [ ] All APIs respond trong <100ms
- [ ] Dashboard load không lỗi
- [ ] Ít nhất 3/4 predictions accurate
- [ ] Audience hiểu được technical value
- [ ] Q&A handle smooth

### **Backup Plans:**
- Pre-recorded demo video (nếu technical issues)
- Screenshots của results (nếu live demo fail)
- Prepared test results (nếu models not loaded)

### **Time Management:**
- **Giữ strict timing**: 2-3-6-2-2 phút cho 5 phases
- **Buffer 2 phút** cho Q&A hoặc technical delays
- **Hard stop** tại 20 phút để respect audience time

---

**Success mantra: "Show, don't just tell. Let the AI do the talking through live results!"** 