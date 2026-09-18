# CreditFlow AI

Alternative credit scoring for Nigeria's informal MSMEs.

Built for the Access Bank × Tobams Group **Next AI Innovators Challenge 2026**.

---

## The problem

Most informal Nigerian businesses — water sachet factories, retail suppliers, local
distributors, provision kiosks — keep their trading records on paper. Daily sales go
into a ruled exercise book; stock purchases are handwritten receipts from the market.
The business is real, the cash flow is real, and the operator often has years of
consistent trading behind them. But because none of it exists as a bank statement,
there is nothing for a credit officer to underwrite against, and the loan application
never starts.

CreditFlow AI turns that paper trail into a credit assessment.

## How it works

1. **The business photographs a page** of its sales logbook or a stack of receipts and
   uploads it.
2. **The document parser reads it.** Azure AI Vision's Read API extracts the text —
   dates, item names, quantities and amounts — and the app suggests a daily revenue
   figure (the median of the Naira amounts on the page, which is robust to one large
   restock line) and a restock amount.
3. **An Access Bank officer reviews and completes the picture.** The figures read from
   the document are presented for confirmation or correction, separately from the
   indicators a single page cannot show — how long the business has traded, how often
   it restocks, how promptly it pays suppliers, how consistent its takings are. The
   officer supplies those from their own knowledge of the business.
4. **The model scores it.** An XGBoost regressor produces a score from 300 to 850, a
   risk tier (Low / Medium / High) and a recommended loan ceiling, with a SHAP
   breakdown showing which factors pushed the score up or down.
5. **The business requests a loan** against its assessment, and the officer approves or
   declines it. An approved request is marked as sent to the bank for disbursement.

This is deliberately **officer-assisted underwriting**, not full automation. The model
reads the document; the officer contributes the relationship context. That division is
visible in the UI, which labels the two groups of inputs separately.

## Status lifecycle

```
pending  →  scored  →  requested  →  approved
                                 └→  rejected
```

## Stack

| Layer | Choice |
|---|---|
| App | Streamlit (single-file router, Access Bank orange `#F15A24` / navy `#0B1F3A`) |
| OCR | Azure AI Vision **Read** API (`azure-ai-vision-imageanalysis`) |
| Scoring | XGBoost regressor, explained with SHAP |
| Database | SQLite — `users`, `submissions`, `profiles`, `sessions` |
| Passwords | PBKDF2-HMAC-SHA256, per-user salt, 100,000 iterations |
| Field encryption | Fernet (`cryptography`) |
| Biometrics (optional) | DeepFace — SFace model, `yunet → ssd → mtcnn` detector fallback |

## Project layout

```
creditflow/
├── app/streamlit_app.py       # the whole UI: auth, MSME pages, officer pages
├── ocr/receipt_parser.py      # Azure Read API + amount/date extraction
├── model/
│   ├── train.py               # trains the regressor
│   ├── predict.py             # score() -> score, risk, max_loan, SHAP explanation
│   └── credit_model.pkl       # trained model (committed — required at runtime)
├── data/
│   ├── generate_synthetic.py  # builds the training set
│   └── synthetic_dataset.csv
├── db.py                      # SQLite schema and all queries
├── crypto_utils.py            # Fernet encrypt/decrypt
├── face_verify.py             # DeepFace wrapper (optional at runtime)
├── schema.py                  # feature names and ranges
├── requirements.txt
├── packages.txt               # apt packages (libgl1 for OpenCV)
└── .streamlit/config.toml     # theme
```

## Running locally

TensorFlow has no wheel for Python 3.14, so the project runs on **Python 3.12**. Use a
single virtual environment for everything.

```bash
python3.12 -m venv venv312
venv312\Scripts\activate          # Windows
# source venv312/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Environment variables

Create a `.env` file in the project root (it is gitignored — never commit it):

```
AZURE_VISION_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_VISION_KEY=<key from the resource's Keys and Endpoint page>
CREDITFLOW_ENCRYPTION_KEY=<a Fernet key>
```

Generate the Fernet key once with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Losing this key makes every encrypted profile field unreadable. Changing it has the
same effect on existing rows.

Optional:

| Variable | Effect |
|---|---|
| `OFFICER_USERNAME` / `OFFICER_PASSWORD` | Credentials for the seeded officer account. Defaults to `Admin` / `Admin`. |
| `DISABLE_FACE_VERIFICATION` | `true` skips the selfie step even where DeepFace is installed. |

## Accounts

Only business owners can register. The officer account is **provisioned, not
self-registered** — `init_db()` seeds exactly one officer on startup, so nobody can
sign themselves up as an approver. Set `OFFICER_USERNAME` / `OFFICER_PASSWORD` in the
environment to control those credentials.

## Deploying to Streamlit Community Cloud

1. Push to GitHub.
2. Create the app, main file path `app/streamlit_app.py`, **Python 3.12** under
   Advanced settings (this cannot be changed after deployment).
3. Paste the environment variables as Cloud **Secrets** in TOML form — `.env` files do
   not exist on Cloud:

   ```toml
   AZURE_VISION_ENDPOINT = "https://<your-resource>.cognitiveservices.azure.com/"
   AZURE_VISION_KEY = "..."
   CREDITFLOW_ENCRYPTION_KEY = "..."
   OFFICER_PASSWORD = "..."
   DISABLE_FACE_VERIFICATION = "true"
   ```

   The app copies secrets into `os.environ` before importing modules that read
   configuration at import time, so no code changes are needed between local and Cloud.

4. `packages.txt` installs `libgl1`, which OpenCV needs on a bare Linux container.

`deepface`, `tensorflow` and `tf-keras` are intentionally **omitted** from
`requirements.txt` for the hosted build — TensorFlow is large enough to strain the free
tier's memory and build limits. The face-verification import is wrapped in a
`try/except`, so the app runs without it and the loan request proceeds without a
selfie. Add those three packages back to run the full flow locally.

## Privacy and access control

Profile fields are treated differently according to who needs them:

- **Encrypted at rest** (Fernet): NIN, account number, date of birth, email, phone
  number, address.
- **Stored in plaintext**: name, business name, gender — an officer needs to read these
  to do their job.
- **Officer view is enforced in SQL, not in the UI.** `get_officer_visible_profile()`
  selects only name, business name, photo, account number and gender. The officer
  screens cannot display the encrypted fields because they are never fetched.

Passwords are hashed with PBKDF2-HMAC-SHA256 (salted, 100,000 iterations) — one-way, so
they cannot be recovered, only reset.

## Known limitations

Stated plainly, because a prototype that hides its edges is harder to evaluate than one
that names them.

- **The model is trained on synthetic data.** `data/generate_synthetic.py` generates the
  training set. The scoring logic is real and discriminates sensibly — a strong trader
  and a failing one score roughly 640 and 380 respectively — but the weights have never
  seen a real Nigerian loan book.
- **Loan ceilings are placeholders.** ₦500,000 / ₦200,000 / ₦50,000 by risk tier are
  illustrative, not derived from Access Bank credit policy.
- **Disbursement is simulated.** Approval changes a status. There is no core-banking or
  payment-rail integration.
- **Phone verification is simulated.** A six-digit code is generated server-side and
  displayed on screen; the UI says so. A production build would deliver it by SMS
  through a provider such as Termii or Africa's Talking.
- **Five of the seven scoring inputs come from the officer**, not the document. Restock
  frequency, POS consistency, supplier payment delay, revenue volatility and months
  operating are relationship facts a single logbook page cannot contain.
- **Streamlit Cloud's disk is not persistent.** Registered users, submissions and
  profiles are lost when the app restarts. The officer account is re-seeded
  automatically; business accounts have to be re-registered.
- **Session tokens live in the URL.** Anyone with the URL is signed in as that user
  until logout. Acceptable for a demo, not for production.
- **No audit logging, and no NDPR-grade compliance work.** A production deployment
  handling real NINs and biometric data would need both.
- **OCR quirks.** The Naira sign is occasionally misread as `$`. Handwriting quality,
  lighting and page angle all affect extraction accuracy, and WhatsApp-forwarded images
  lose the EXIF orientation data used to correct rotation.

## Credits

Akinbode John Iyanuoluwa — Access Bank × Tobams Group Next AI Innovators Challenge 2026.
