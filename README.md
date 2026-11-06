# Лабораторна робота №02

## LLM-екстракція знань, векторний пошук та GraphRAG

## 1. Предметна область і корпус даних

Предметна область роботи — ноутбуки Googlebook та пов'язана з ними
екосистема: Googlebook OS, Android, Gemini, апаратні компоненти,
виробники ноутбуків і конкуренти.

Корпус складається з 8 зв'язних текстових документів у каталозі
[`corpus/`](./corpus/). Загальний обсяг корпусу — приблизно 7738 слів.
Тексти описують:

- концепцію та призначення Googlebook;
- запуск і позиціонування продукту;
- моделі Lenovo, Acer, Asus, Dell та HP;
- процесори, дисплеї та інші апаратні компоненти;
- інтеграцію з Android-смартфонами;
- функції Gemini та Googlebook OS;
- порівняння з Chromebook, MacBook Air і Surface Laptop.

### Джерела даних

Документи були зібрані з відкритих вебпублікацій про Googlebook і
підготовлені як локальний навчальний корпус. Для остаточного звіту потрібно
вказати точні URL та бібліографічні дані:

| Файл    | URL                                                                                                                  |
| ------- | -------------------------------------------------------------------------------------------------------------------- |
| `1.txt` | [https://blog.google/products-and-platforms/devices/googlebook/pre-order-googlebook/]                                |
| `2.txt` | [https://en.wikipedia.org/wiki/Googlebook]                                                                           |
| `3.txt` | [https://news.lenovo.com/pressroom/press-releases/first-googlebook-premium-ai-experiences-sleek-lightweight-design/] |
| `4.txt` | [https://techcrunch.com/2026/09/21/googles-899-googlebook-is-a-bet-that-youll-buy-a-new-laptop-for-gemini/]          |
| `5.txt` | [https://www.theguardian.com/technology/2026/oct/20/googlebook-review-google-laptops-chromeos-android]               |
| `6.txt` | [https://tech.yahoo.com/computing/articles/google-next-googlebook-could-android-091326536.html]                      |
| `7.txt` | [https://mezha.ua/articles/google-googlebook-start-315469/]                                                          |
| `8.txt` | [https://9to5google.com/2026/09/21/googlebook-hands-on-what-makes-a-laptop-a-laptop-video/]                          |

### Питання до корпусу

#### Прості питання

Відповідь на кожне питання безпосередньо наведена в одному документі:

1. According to the Lenovo Googlebook 15 specification, which processor is
   listed?
2. Which feature allows the Acer Googlebook 14 to fold into different
   positions?
3. What starting price is stated for Googlebook pre-orders?

#### Синтетичні питання

Відповідь потребує зіставлення інформації з кількох документів:

4. Which companies manufacture Googlebook devices, and which processors are
   associated with their models?
5. Which Googlebook features repeatedly connect the laptop with Android phones
   and Google services?
6. How is Googlebook positioned against Chromebooks, MacBook Air, and Surface
   Laptop?

Глобальне питання для порівняння GraphRAG і vector-RAG:

> Which companies, products and technologies are central to the Googlebook
> story, and how are they connected?

## 2. Схема сутностей та LLM-екстракція

Схему описано в [`schema.py`](./schema.py). Використано 6 типів сутностей:

```text
Product, Company, Hardware, Feature, Software, Chip
```

та 7 типів зв'язків:

```text
MADE_BY, POWERED_BY, RUNS, HAS_FEATURE,
USES_HARDWARE, REPLACES, COMPETES_WITH
```

Приклади видобутих фактів:

```text
Googlebook MADE_BY Acer
Googlebook MADE_BY Lenovo
Googlebook POWERED_BY Snapdragon X Elite
Lenovo Googlebook 15 POWERED_BY Intel Core Ultra 5
Googlebook RUNS Googlebook OS
Googlebook HAS_FEATURE Gemini Intelligence
Googlebook USES_HARDWARE NVMe storage
Googlebook REPLACES Chromebook
Googlebook COMPETES_WITH MacBook Air
```

Екстракція виконується скриптом [`extract.py`](./extract.py) через Gemini
structured output. Результат зберігається у [`facts.json`](./facts.json).
У поточному запуску отримано 79 унікальних фактів.

### Ручна перевірка 20 фактів

Випадкову вибірку сформовано з фіксованим seed `42`. Її потрібно перевірити
за оригінальними документами та заповнити таблицю:

|   № | Факт                                                                  | Документ | Правильний? | Коментар                                                                                                     |
| --: | --------------------------------------------------------------------- | -------- | ----------- | ------------------------------------------------------------------------------------------------------------ |
|   1 | Googlebook REPLACES Chromebook                                        | `2.txt`  | Так         | У тексті Googlebook названо наступником Chromebook.                                                          |
|   2 | Googlebook MADE_BY HP                                                 | `1.txt`  | Так         | HP наведено серед виробників стартових пристроїв.                                                            |
|   3 | Googlebook HAS_FEATURE vibe-coded widgets                             | `4.txt`  | Так         | Функція явно згадана в описі Googlebook.                                                                     |
|   4 | Lenovo Googlebook 15 USES_HARDWARE Lenovo AI Mouse                    | `3.txt`  | Частково    | Аксесуар спеціально розроблений для Googlebook, але `USES_HARDWARE` не є найточнішим зв'язком.               |
|   5 | Lenovo Googlebook 15 HAS_FEATURE fingerprint reader                   | `3.txt`  | Так         | Інтегрований сканер відбитка явно згаданий.                                                                  |
|   6 | Googlebook USES_HARDWARE Arm64 SoC                                    | `2.txt`  | Так         | Arm64 SoC прямо наведено в описі платформи.                                                                  |
|   7 | Googlebook MADE_BY Google                                             | `2.txt`  | Ні          | Google створила платформу, але виробниками пристроїв названо OEM-партнерів                                   |
|   8 | Googlebook HAS_FEATURE Google Play Store                              | `7.txt`  | Так         | Google Play Store прямо згаданий серед можливостей ноутбуків.                                                |
|   9 | Googlebook HAS_FEATURE Create My Widget                               | `1.txt`  | Так         | Функція описана окремим пунктом.                                                                             |
|  10 | Acer Googlebook 14 USES_HARDWARE fingerprint readers                  | `5.txt`  | Ні          | У тексті fingerprint readers наведено як загальний стандарт лінійки, без прямого прив'язування саме до Acer. |
|  11 | Googlebook MADE_BY Lenovo                                             | `1.txt`  | Так         | Lenovo наведено серед виробників стартових пристроїв.                                                        |
|  12 | Acer Googlebook 14 HAS_FEATURE matte display                          | `8.txt`  | Так         | Матове покриття Acer Googlebook 14 описано безпосередньо.                                                    |
|  13 | Dell XPS Googlebook MADE_BY Dell                                      | `8.txt`  | Так         | У матеріалі згадується модель Dell XPS Googlebook.                                                           |
|  14 | Lenovo Googlebook 15 HAS_FEATURE quad-speaker system with Dolby Atmos | `3.txt`  | Так         | Характеристика прямо наведена у специфікації моделі.                                                         |
|  15 | Lenovo Googlebook 15 HAS_FEATURE facial authentication                | `3.txt`  | Так         | Функція розпізнавання обличчя прямо згадана.                                                                 |
|  16 | Googlebook USES_HARDWARE Titan C                                      | `7.txt`  | Так         | Titan C описано як модуль безпеки Googlebook.                                                                |
|  17 | Googlebook HAS_FEATURE lightbar                                       | `8.txt`  | Так         | Lightbar описано як елемент лінійки Googlebook.                                                              |
|  18 | Googlebook USES_HARDWARE Google Titan                                 | `1.txt`  | Так         | Google Titan hardware root of trust прямо згаданий.                                                          |
|  19 | Acer Googlebook 14 RUNS Googlebook OS                                 | `8.txt`  | Так         | Acer Googlebook 14 входить до ноутбуків, що працюють під Googlebook OS.                                      |
|  20 | Acer Googlebook 14 MADE_BY Acer                                       | `5.txt`  | Так         | Модель прямо названа Acer Googlebook 14.                                                                     |

Результат перевірки:

```text
Правильних фактів: 17 із 20
Частка безумовно правильних: 85%
Частково правильних: 1 із 20 (5%)
Помилкових: 2 із 20 (10%)
```

Типові помилки моделі:

- неточний зв'язок `USES_HARDWARE` для аксесуара Lenovo AI Mouse;
- неправильне використання `MADE_BY` для Google: джерело підтверджує
  розробку платформи, а не виготовлення пристрою Google;
- неправильне зв'язування загальної характеристики `fingerprint readers`
  саме з моделлю Acer Googlebook 14.

## 3. Векторний пошук

Корпус розбивається на чанки в [`common.py`](./common.py). Спочатку текст
ділиться за абзацами, а абзаци довші за 700 символів — за реченнями.
У результаті отримано 169 чанків.

Для embedding використовується модель `gemini-embedding-001`.

Пошук реалізовано у [`vector_search.py`](./vector_search.py):

- `search_keyword()` — TF-IDF keyword-пошук;
- `search_vector()` — семантичний пошук за Gemini embeddings.

### Порівняння методів

Порівняння виконано для шести запитів із [`vector_search.py`](./vector_search.py).
У таблиці наведено найважливіший результат із top-3:

| Запит | Keyword-пошук | Semantic-пошук | Висновок |
|---|---|---|---|
| Процесор Lenovo Googlebook 15 | `3#3` — знайшов назву моделі, але не процесор | `3#2` — знайшов `Intel Core Ultra 5` | Semantic кращий |
| Складаний екран Acer | `5#5` — знайшов прямий опис folding screen | `6#7`, `5#5` — релевантний чанк про Acer на другому місці | Keyword трохи кращий |
| Стартова ціна Googlebook | Знайшов згадки про pre-orders, але не ціну | `1#26` — знайшов `$899` | Semantic кращий |
| Виробники та процесори | Повернув загальні фрагменти без повного списку | Повернув частково релевантні фрагменти про моделі й SoC | Для синтезу краще GraphRAG |
| Інтеграція з Android | Повернув загальні згадки про Android | `1#12`, `1#14`, `1#13` — точні фрагменти про інтеграцію | Semantic значно кращий |
| Позиціонування проти конкурентів | `5#0` — знайшов матеріал про конкуренцію | `5#0` та додатковий контекст про ціну й позиціонування | Semantic трохи повніший |

Приклад переваги semantic search: запит може описувати інтеграцію ноутбука
зі смартфоном іншими словами, без точного збігу всіх слів із текстом.

Приклад переваги keyword search: пошук точної назви процесора, наприклад
`Snapdragon X Elite`, коду або числового позначення моделі.

У цьому експерименті найкращим прикладом переваги semantic search є запит
про Android-інтеграцію: TF-IDF повертає загальні згадки, а semantic search
знаходить фрагменти про продовження роботи між телефоном і ноутбуком,
синхронізацію налаштувань та доступ до файлів телефона.

Для точного пошуку keyword-підхід може бути кращим. Наприклад, запит
`Snapdragon X Elite` шукає конкретну назву процесора, тому буквальний збіг
надійніший за семантичну близькість до інших процесорів.

## 4. Граф знань і GraphRAG

Граф будується у [`graphrag.py`](./graphrag.py) на основі `facts.json`:

- вузли — сутності;
- ребра — зв'язки між сутностями;
- алгоритм Louvain — виявлення спільнот;
- Gemini — резюме кожної спільноти та фінальний синтез.

У поточному запуску:

- 69 вузлів;
- 79 ребер;
- 1 зв'язна компонента;
- 5 спільнот Louvain.

Візуалізація збережена у
[`knowledge_graph.png`](./knowledge_graph.png).

### GraphRAG-відповідь

GraphRAG узагальнив, що Googlebook є спільною екосистемою Google та
виробників ASUS, Acer, Dell, HP і Lenovo. У ній поєднуються Googlebook OS
або Android OS, Gemini та інші AI-функції, процесори Intel, Qualcomm і
MediaTek, а також апаратні компоненти та інтеграція з Android-смартфонами.
Googlebook позиціонується як наступник Chromebook і конкурент MacBook Air та
Surface Laptop.

### Vector-RAG-відповідь

Vector-RAG на основі шести найбільш релевантних чанків сформулював коротшу
відповідь: Googlebook — це комп'ютерна платформа з Googlebook OS, Arm64 або
x64 SoC, UEFI boot і NVMe storage, орієнтована також на користувачів
Android-смартфонів.

### Порівняння

GraphRAG краще підходить для глобального питання, оскільки спочатку групує
сутності та зв'язки у спільноти, а потім синтезує інформацію з усіх груп.
Тому він знаходить зв'язки між компаніями, моделями, процесорами,
програмним забезпеченням і конкурентами.

Vector-RAG повертає лише кілька найближчих текстових фрагментів. Він добре
відповідає на локальні питання, але може не побачити зв'язки, розподілені
між різними документами.

## 5. Помилковий і виправлений приклад

### Помилка під час виконання

Початковий запуск векторного індексу завершився помилкою:

```text
400 INVALID_ARGUMENT:
BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch
```

Причина — корпус містив 169 чанків, але всі вони передавалися до Gemini
одним запитом.

### Виправлення

У [`common.py`](./common.py) embedding-запити було розділено на пакети
максимум по 100 текстів. Також додано паузу між пакетами для дотримання
обмежень безкоштовного рівня API. Після виправлення файли
[`chunks.json`](./chunks.json) і [`embeddings.npy`](./embeddings.npy) були
успішно створені.

Інша помилка виникла через кодування консолі Windows: український `print`
викликав `UnicodeEncodeError`. Для stdout скриптів встановлено UTF-8.

## 6. Висновки

У роботі побудовано наскрізний pipeline:

```text
тексти → LLM-екстракція → структуровані факти →
embeddings → векторний пошук → граф знань → GraphRAG
```

LLM structured output дає змогу перетворити неструктурований корпус на
множину фактів із заданою схемою. Семантичний пошук краще працює із
перефразованими запитами, а TF-IDF — із точними назвами та кодами.
GraphRAG дає повнішу відповідь на глобальні питання, коли потрібне
узагальнення зв'язків між кількома документами.
