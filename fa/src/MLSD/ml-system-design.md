# <a name="ml-sys"></a> ۴. طراحی سیستم یادگیری ماشین

| بخش |
| --- |
| ۱. [فرمول ۹ مرحله‌ای طراحی سیستم ML](#fa-9-step-formula) |
| ۲. [نمونه سؤال‌های طراحی سیستم ML](#fa-sample-questions) |
| ۳. [موضوع‌های طراحی سیستم ML](#fa-system-design-topics) |
| ۴. [سیستم ML در شرکت‌های بزرگ فناوری](#fa-big-tech-ml) |
| ۵. [طراحی سیستم هوش مصنوعی عامل‌محور](https://github.com/alirezadir/Agentic-AI-Systems.git) |

### طراحی سیستم ML برای production

استقرار مدل یادگیری عمیق در production فقط به آموزش مدلی با performance خوب محدود نمی‌شود. برای ساخت یک سیستم یادگیری عمیق production-level باید چند جزء مستقل را طراحی، پیاده‌سازی و به یکدیگر متصل کنید.

<p align="center">
<img src="../../../src/imgs/components.png" title="" width="90%" height="80%">
</p>

حل مسئلهٔ طراحی سیستم ML از نظر منطقی شبیه طراحی سیستم نرم‌افزاری عمومی است. برای مرور مبانی می‌توانید [Grokking the System Design Interview](https://www.designgurus.io/course/grokking-the-system-design-interview) و [System Design Primer](https://github.com/donnemartin/system-design-primer) را ببینید.

با این حال، یک سیستم ML اجزایی دارد که به توجه ویژه نیاز دارند؛ از داده و feature تا آموزش، inference، ارزیابی و monitoring.

### مصاحبهٔ طراحی سیستم ML

- سؤال‌ها open-ended هستند و معمولاً یک پاسخ کاملاً درست و یکتا ندارند.
- هدف، سنجش توانایی شما در دیدن تصویر کلان و طراحی یک سیستم ML در سطح production است که بتوان آن را به‌عنوان سرویس در زیرساخت شرکت مستقر کرد.

# <a id="fa-9-step-formula"></a> ۱. فرمول ۹ مرحله‌ای طراحی سیستم ML

برای کاربرد واقعی باید جریان طراحی مشخصی داشته باشید. فرمول **۹ مرحله‌ای طراحی سیستم ML** چارچوبی برای مسئله‌های کسب‌وکار مرتبط با ML در محیط کار و مصاحبه است. [template انگلیسی](../../../src/MLSD/mlsd-template.md) نیز در دسترس است.

| مرحله | موضوع |
| --- | --- |
| ۱ | [صورت‌بندی مسئله](#fa-step-1) |
| ۲ | [معیارهای آفلاین و آنلاین](#fa-step-2) |
| ۳ | [اجزای معماری و منطق MVP](#fa-step-3) |
| ۴ | [گردآوری و آماده‌سازی داده](#fa-step-4) |
| ۵ | [مهندسی ویژگی](#fa-step-5) |
| ۶ | [توسعه و ارزیابی آفلاین مدل](#fa-step-6) |
| ۷ | [سرویس پیش‌بینی](#fa-step-7) |
| ۸ | [آزمون آنلاین و استقرار](#fa-step-8) |
| ۹ | [مقیاس‌دهی، پایش و به‌روزرسانی](#fa-step-9) |

در مصاحبه انعطاف‌پذیر باشید. بسته به نیاز سؤال و علاقهٔ مصاحبه‌کننده می‌توانید بعضی بخش‌ها را سریع‌تر رد کنید یا روی یک یا دو بخش deep dive انجام دهید.

## <a id="fa-step-1"></a> ۱. صورت‌بندی مسئله

- سؤال‌های روشن‌کننده بپرسید.
- use case و هدف کسب‌وکار را مشخص کنید.
- requirementها را تعیین کنید:
  - scope و featureهای لازم، scale و personalization؛
  - latency پیش‌بینی و تعداد predictionها؛
  - constraintها؛
  - منبع و دسترس‌پذیری داده.
- فرض‌ها را شفاف بنویسید.
- مسئلهٔ انتزاعی را به مسئلهٔ ML تبدیل کنید:
  - objective مدل؛
  - ورودی و خروجی؛
  - دستهٔ مسئله، مانند binary/multiclass classification یا unsupervised learning.
- آیا واقعاً به ML نیاز داریم؟
  - اثر مورد انتظار را با هزینهٔ گردآوری/برچسب‌گذاری داده و compute مقایسه کنید.
  - اگر پاسخ منفی است، طراحی سیستم عمومی مناسب‌تر است. در مصاحبهٔ طراحی سیستم ML معمولاً می‌توان نیاز به ML را فرض کرد.

## <a id="fa-step-2"></a> ۲. معیارهای آفلاین و آنلاین

- معیارهای آفلاین
  - classification: precision، recall، F1، ROC-AUC، PR-AUC، mAP و log-loss؛ نامتوازنی class را در نظر بگیرید.
  - retrieval و ranking: Precision@k و Recall@k کیفیت ترتیب را نمی‌سنجند؛ mAP، MRR و nDCG ترتیب را نیز لحاظ می‌کنند.
  - regression: MSE و MAE
  - معیار خاص مسئله: BLEU، BLEURT، GLUE و ROUGE برای زبان یا CPE برای تبلیغات
  - latency و هزینهٔ محاسباتی، به‌ویژه روی device
- معیارهای آنلاین
  - CTR، نرخ موفقیت task/session و کل زمان session
  - engagement مانند like/comment rate
  - conversion، revenue lift و reciprocal rank نخستین click
  - counter metric و feedback منفی مستقیم مانند hide یا report
- trade-off میان معیارها را توضیح دهید؛ بهینه‌کردن یک معیار ممکن است دیگری را بدتر کند.

## <a id="fa-step-3"></a> ۳. اجزای معماری و منطق MVP

- معماری سطح بالا و اجزای اصلی
  - اجزای غیر ML مانند user، app server، database و knowledge graph و تعامل آن‌ها
  - اجزای ML مانند candidate generator، ranker و مولد دادهٔ آموزش
- معماری modular
  - مدل نخست، برای مثال candidate generation
  - مدل دوم، مانند ranker یا filter
  - interface، ورودی و خروجی هر module

## <a id="fa-step-4"></a> ۴. گردآوری و آماده‌سازی داده

- نیاز داده: target، actorهای اصلی مانند user/item، نوع داده مانند متن/تصویر/ویدئو و حجم
- منبع داده: دسترس‌پذیری و هزینه؛ implicit از logging یا explicit از survey کاربر
- محل و سیاست ذخیره‌سازی
- نوع داده
  - structured: عددی گسسته/پیوسته و categorical ترتیبی/اسمی
  - unstructured: تصویر، متن، ویدئو و صوت
- برچسب‌گذاری در مسئلهٔ supervised
  - label طبیعی مانند click، like یا purchase
  - نبود label منفی؛ clickنکردن الزاماً منفی نیست و ممکن است negative sampling لازم باشد.
  - feedback صریح کاربر
  - human annotation که گران، کند و دارای ملاحظهٔ privacy است
- کمبود label
  - programmatic labeling با هزینهٔ کمتر و سازگاری بیشتر، اما همراه با noise
  - semi-supervised learning از مجموعهٔ کوچک اولیه
  - weak supervision با heuristic، keyword، regex، database یا خروجی مدل دیگر
  - transfer learning: pretrain روی دادهٔ بزرگ و ارزان، سپس zero-shot یا fine-tune روی downstream task
  - active learning
- data augmentation
- پایپ‌لاین تولید داده
  - ingestion آفلاین یا آنلاین
  - تولید و transform ویژگی
  - تولید label
  - joinکردن داده‌ها

## <a id="fa-step-5"></a> ۵. مهندسی ویژگی

- انتخاب ویژگی
  - actorهای اصلی مانند user، item، document، query، ad و context
  - ویژگی فعلی و تاریخی هر actor؛ مانند profile، history و interest کاربر
  - ویژگی متن مانند n-gram، intent، topic، frequency، length و embedding
  - cross feature مانند user-item یا query-document؛ برای مثال TF-IDF یا سابقهٔ تعامل user با video/ad
  - constraintهای privacy
- نمایش ویژگی
  - one-hot encoding
  - embedding متن، تصویر، graph، user یا store؛ روش یادگیری، precompute و storage
  - encoding categorical با one-hot، ordinal یا count
  - positional embedding
  - scaling و normalization ویژگی عددی
- پیش‌پردازش دادهٔ unstructured
  - متن: normalization، pre-tokenization، tokenizer در سطح character/word/subword و افزودن token ویژه
  - تصویر: resize و normalize
  - ویدئو: decode frame، sampling، resize، scaling و normalization
- مقدار گمشده، اهمیت ویژگی و featurizer تبدیل raw data به feature
- تفاوت feature استاتیک در feature store با feature دینامیک محاسبه‌شده به‌صورت آنلاین

## <a id="fa-step-6"></a> ۶. توسعه و ارزیابی آفلاین مدل

- انتخاب مدل برای MVP
  - مسیر مناسب: heuristic، سپس مدل ساده، مدل پیچیده‌تر و در صورت نیاز ensemble
  - با ساده‌ترین راه‌حل ممکن شروع کنید و iterate کنید؛ اصل KISS.
  - گزینه‌های رایج: رگرسیون لجستیک، درخت تصمیم، GBDT/XGBoost، Random Forest، شبکهٔ feedforward، CNN، RNN و Transformer
  - عوامل تصمیم: پیچیدگی task، نوع/حجم/پیچیدگی داده، سرعت آموزش، compute/latency/memory در inference، continual learning و interpretability
  - [معماری‌های رایج شبکهٔ عصبی](../../../src/MLSD/mlsd-modeling-popular-archs.md)
- مجموعه‌داده
  - sampling غیراحتمالی یا احتمالی، مانند random، stratified، reservoir و importance sampling
  - splitهای train/dev/test و نسبت آن‌ها
  - برای دادهٔ وابسته به زمان، split زمانی و اثر seasonality/trend
  - جلوگیری از data leakage: scaling پس از split و استفاده از train برای آمار، scaling و مقدار گمشده
  - نامتوازنی class: resampling، loss وزن‌دار یا ترکیب classها
- آموزش مدل
  - lossهایی مانند MSE، binary/categorical cross-entropy، MAE، Huber، hinge و contrastive
  - optimizerهایی مانند SGD، AdaGrad، RMSProp و Adam
  - آموزش از صفر یا fine-tuning
  - validation، debugging و تفاوت آموزش آفلاین/آنلاین
  - ارزیابی آفلاین
  - hyperparameter tuning مانند grid search
  - iteration روی مدل MVP، data augmentation و فرکانس update
  - calibration مدل

## <a id="fa-step-7"></a> ۷. سرویس پیش‌بینی

- پردازش و اعتبارسنجی داده
- web app، serving system و prediction service
- prediction از نوع batch، online یا hybrid
  - batch: دوره‌ای، از پیش محاسبه و ذخیره‌شده، با throughput بالا
  - online: محاسبه هنگام رسیدن request، با latency کم
  - hybrid: برای مثال Netflix می‌تواند titleها را batch و rowها را online بسازد.
- سرویس nearest neighbor و ANN با tree، LSH یا clustering
- ML روی edge یا on-device AI
  - مزایا: کاهش وابستگی به network/latency، privacy و هزینه
  - محدودیت: حافظه، compute و انرژی
  - فشرده‌سازی مدل با quantization، pruning، knowledge distillation و factorization

## <a id="fa-step-8"></a> ۸. آزمون آنلاین و استقرار مدل

- آزمایش A/B
  - سهم کاربران، گروه control/test و null hypothesis
- bandit
- shadow deployment
- canary release

## <a id="fa-step-9"></a> ۹. مقیاس‌دهی، پایش و به‌روزرسانی

- scaleکردن تقاضا مانند سیستم توزیع‌شده: server، load balancer، sharding، replication و cache؛ partition دادهٔ آموزش یا knowledge base
- scaleکردن ML
  - distributed ML با data parallelism و model parallelism در train/inference
  - synchronous و asynchronous SGD
  - distributed training مبتنی بر data parallel یا RPC
  - scaleکردن گردآوری داده؛ مانند [ترجمهٔ ماشینی برای ۱۰۰۰ زبان](https://arxiv.org/abs/2205.03983) و [NLLB](https://research.facebook.com/publications/no-language-left-behind/)
  - AutoML از hyperparameter tuning تا architecture search یا NAS
- monitoring
  - logging ویژگی، prediction، معیار و event
  - معیارهای نرم‌افزار و ML، dashboard آفلاین و آنلاین
  - تغییر توزیع داده: covariate shift، label shift و concept shift؛ تشخیص آماری و اصلاح
- failure
  - نرم‌افزار: dependency، deployment، hardware و downtime
  - ML: تفاوت test و online، feedback loop، ورودی نامعتبر و تغییر توزیع
  - alarm برای شکست پایپ‌لاین، آموزش، deployment یا افت معیار
- update و continual training
  - آموزش از صفر یا base model و انتخاب cadence روزانه، هفتگی یا ماهانه
  - auto-update، active learning و human-in-the-loop ML

### موضوع‌های تکمیلی

- iteration روی طراحی پایه برای افزودن feature جدید
- بایاس دادهٔ آموزش و بایاس ناشی از human labeling
- freshness و diversity
- privacy و security

# <a id="fa-sample-questions"></a> ۲. نمونه سؤال‌های طراحی سیستم ML

### سیستم‌های هوش مصنوعی مولد و LLM در سال ۲۰۲۶

این موضوع‌ها بیشتر به‌صورت مرحله‌ای مستقل با نام طراحی سیستم هوش مصنوعی مولد و LLM مطرح می‌شوند. همان فرمول ۹ مرحله‌ای با نگاه هوش مصنوعی مولد کاربرد دارد. بخش [طراحی سیستم هوش مصنوعی مولد و LLM](#fa-genai-llm-system-design) و مخزن [سیستم‌های هوش مصنوعی عامل‌محور](https://github.com/alirezadir/Agentic-AI-Systems.git) را ببینید.

- Q&A اسناد با RAG یا «گفت‌وگو با اسناد»
- chatbot پشتیبانی مشتری مبتنی بر LLM با guardrail و انتقال به انسان
- workflow عامل‌محور یا دستیار AI با planning، tool use و memory
- semantic/enterprise search همراه با synthesis و citation
- دستیار کدنویسی یا coding agent با RAG روی repository، ابزار و verification
- تولید محتوا یا summarization در scale بالا با batch و safety filtering
- recommendation و personalization مبتنی بر LLM به‌عنوان ranker یا feature generator

### سیستم‌های پیشنهاددهنده

- [پیشنهاد ویدئو یا فیلم](../../../src/MLSD/mlsd-video-recom.md) برای Netflix/YouTube
- [پیشنهاد دوست یا follower](../../../src/MLSD/mlsd-pymk.md) برای Facebook/Twitter/LinkedIn
- [پیشنهاد event](../../../src/MLSD/mlsd-event-recom.md)، [بازی](../../../src/MLSD/mlsd-game-recom.md)، محصول جایگزین، اجاره یا مکان

### جست‌وجو و رتبه‌بندی

- جست‌وجوی document با [query متنی](../../../src/MLSD/mlsd-search.md)، [تصویر/ویدئو](../../../src/MLSD/mlsd-image-search.md) یا [query چندوجهی](../../../src/MLSD/mlsd-mm-video-search.md)
- [رتبه‌بندی newsfeed](../../../src/MLSD/mlsd-newsfeed.md)
- serving تبلیغ و [پیش‌بینی click تبلیغ](../../../src/MLSD/mlsd-ads-ranking.md)

### NLP، CV و خودرو خودران

- entity linking، typeahead، sentiment، language identification، chatbot و question answering
- image blurring و OCR
- خودروی خودران: perception، prediction، planning و [تشخیص عبور عابر](../../../src/MLSD/mlsd-av.md)
- ride matching

### موارد دیگر

- proximity service، تخمین زمان تحویل غذا و diagnosis پزشکی
- تشخیص محتوای مضر، spam یا fraud؛ از جمله [تشخیص چندوجهی محتوای مضر](../../../src/MLSD/mlsd-harmful-content.md)

# <a id="fa-system-design-topics"></a> ۳. موضوع‌های طراحی سیستم ML

### سیستم پیشنهاددهنده

- candidate generation با collaborative filtering مبتنی بر user/item، matrix factorization، two-tower یا content-based filtering
- ranking و learning-to-rank از نوع pointwise، pairwise یا listwise

### جست‌وجو و ranking

- keyword و [semantic search](https://txt.cohere.ai/what-is-semantic-search/?utm_source=linkedin&utm_medium=paidsocial&utm_campaign=contentpromotion_bloglookalikes)، visual search و video search
- معماری دو مرحله‌ای برای document selection و ranking
- ranking برای newsfeed و ad؛ classification، multi-stage ranking، blender و filter

### پردازش زبان طبیعی

- پیش‌پردازش، tokenization و feature engineering
- text embedding با Word2Vec، GloVe، ELMo و BERT
- طبقه‌بندی متن، sentiment و topic modeling
- sequence tagging با NER، POS، HMM، Viterbi و beam search
- تولید متن و language modeling؛ trade-off n-gram با deep learning و روش decoding
- seq2seq، machine translation، NMT و Transformer
- question answering، dialog و chatbot؛ [درس CMU دربارهٔ chatbot](http://tts.speech.cs.cmu.edu/courses/11492/slides/chatbots_shrimai.pdf) و [spoken dialogue system](http://tts.speech.cs.cmu.edu/courses/11492/slides/sds_components.pdf)
- speech recognition با MFCC، acoustic model، HMM و CTC

### بینایی کامپیوتر

- image classification با VGG و ResNet
- [object detection](https://viso.ai/deep-learning/object-detection/) با R-CNN/Fast/Faster R-CNN یا YOLO/SSD
- [Vision Transformer](https://viso.ai/deep-learning/vision-transformer-vit/) و NMS
- object tracking

### مسئله‌های graph

- People You May Know

### <a id="fa-genai-llm-system-design"></a> طراحی سیستم هوش مصنوعی مولد / LLM در سال ۲۰۲۶

این مهم‌ترین افزودهٔ مصاحبهٔ مدرن طراحی سیستم ML است. در سال ۲۰۲۶، «روش ارزیابی، طراحی سیستم جدید است». مصاحبه‌کننده بیش از diagram معماری به **هزینه، latency، guardrail و monitoring** توجه می‌کند. باید بدانید LLM کجا مفید است و کجا سیستم deterministic باید کنترل را در دست بگیرد.

- **راهبرد دانش و context:** انتخاب میان RAG، fine-tuning، context طولانی و ابزار/memory
  - RAG برای دانش بزرگ، متغیر یا نیازمند citation؛ fine-tuning برای رفتار، format و style دامنه؛ context طولانی برای reasoning روی سند کامل. در عمل می‌توان instruction-tuned base، LoRA سبک و RAG را ترکیب کرد.
- **پایپ‌لاین RAG**
  - ingestion: chunking و overlap، embedding model و vector DB/ANN مانند HNSW یا IVF-PQ
  - retrieval: dense یا hybrid با BM25، metadata filter، reranking با cross-encoder و query rewriting/HyDE
  - generation: prompt grounded، citation، بودجهٔ context window و مدیریت «پاسخی پیدا نشد»
  - حالت پیشرفته: RAG چندمرحله‌ای یا عامل‌محور، GraphRAG و cache برای prompt/embedding/semantic result
- **سیستم عامل‌محور:** planning با ReAct یا plan-and-execute، tool/function calling، memory کوتاه‌مدت/بلندمدت، orchestration چند عامل، recovery از loop و ابزار hallucinated و human-in-the-loop
- **serving و scaling:** طراحی inference API، KV cache، continuous batching در vLLM، quantization، speculative decoding، routing میان مدل کوچک/بزرگ و trade-off self-host با API
- **قابلیت اطمینان:** rate limit، retry، timeout، idempotency، fallback، queue، circuit breaker و graceful degradation
- **guardrail و safety:** فیلتر ورودی/خروجی، دفاع در برابر prompt injection و jailbreak، حذف PII، بررسی grounding/hallucination، moderation و refusal
- **ارزیابی:** golden set، LLM-as-judge، win-rate جفتی، RAG triad، recall@k/MRR/nDCG و ترکیب offline eval، A/B آنلاین و regression test پیش از انتشار
- **هزینه و latency:** شمارش token، بهینه‌سازی prompt/context، caching، batching، streaming و تفاوت TTFT با latency کل، و routing به مدل کوچک
- **monitoring:** quality drift، نرخ hallucination، retrieval hit rate، latency/cost هر request، feedback loop و trace مرحله‌به‌مرحلهٔ agent

# <a id="fa-big-tech-ml"></a> ۴. سیستم ML در شرکت‌های بزرگ فناوری

پس از یادگیری مبانی، وبلاگ‌های مهندسی شرکت‌ها دربارهٔ سیستم‌های ML را مطالعه کنید. تعدادی از این منابع در بخش [ML at Companies](../../../src/MLSD/ml-companies.md) گردآوری شده‌اند.

# منابع بیشتر

- [Full Stack Deep Learning](https://fall2019.fullstackdeeplearning.com/)
- [Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
- [دورهٔ Stanford دربارهٔ طراحی سیستم ML](https://online.stanford.edu/courses/cs329s-machine-learning-systems-design)
