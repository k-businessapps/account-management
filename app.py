import re
import json
from io import BytesIO
from datetime import date
import pandas as pd
import numpy as np
import requests
import streamlit as st

try:
    from dateutil.relativedelta import relativedelta
except Exception:
    relativedelta = None


APP_TITLE = "Account Management"
APP_SUBTITLE = "Upsells and churn, aligned to renewal validity and month-end cutoffs"


# KrispCall palette (official)
KC_LIGHT_PINKISH_PURPLE = "#F4B7FF"
KC_VIBRANT_MAGENTA = "#EA66FF"
KC_BRIGHT_VIOLET = "#8548FF"
KC_DEEP_PURPLE = "#8D34F0"
KC_WHITE = "#FFFFFF"
KC_LIGHT_GRAY = "#EFEFEF"
KC_TEXT = "#15151A"


def _require_secrets():
    missing = []
    if "mixpanel" not in st.secrets:
        missing.append("mixpanel")
    else:
        for k in ["project_id", "auth_header", "from_date"]:
            if k not in st.secrets["mixpanel"]:
                missing.append(f"mixpanel.{k}")

    if "auth" not in st.secrets:
        missing.append("auth")
    else:
        for k in ["username", "password"]:
            if k not in st.secrets["auth"]:
                missing.append(f"auth.{k}")

    if missing:
        st.error(
            "Missing required secrets. Add these keys in .streamlit/secrets.toml.\n\n"
            + "\n".join(f"- {m}" for m in missing)
        )
        st.stop()



def inject_brand_css():
    css = f"""
    <style>
      :root {{
        --kc-light: {KC_LIGHT_PINKISH_PURPLE};
        --kc-magenta: {KC_VIBRANT_MAGENTA};
        --kc-violet: {KC_BRIGHT_VIOLET};
        --kc-deep: {KC_DEEP_PURPLE};
        --kc-bg: {KC_WHITE};
        --kc-muted: {KC_LIGHT_GRAY};
        --kc-text: {KC_TEXT};
        --radius: 16px;
      }}

      .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.2rem;
        max-width: 1220px;
      }}

      .stApp {{
        background: var(--kc-bg);
        color: var(--kc-text);
      }}

      section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(133,72,255,0.10), rgba(244,183,255,0.10));
        border-right: 1px solid rgba(21,21,26,0.06);
      }}

      .kc-header {{
        display: flex;
        gap: 14px;
        align-items: center;
        padding: 14px 16px;
        border-radius: var(--radius);
        background: linear-gradient(90deg, rgba(141,52,240,0.10), rgba(234,102,255,0.10), rgba(133,72,255,0.10));
        border: 1px solid rgba(21,21,26,0.06);
        margin-bottom: 14px;
      }}
      .kc-title {{
        font-size: 1.25rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
      }}
      .kc-subtitle {{
        font-size: 0.95rem;
        margin: 2px 0 0 0;
        opacity: 0.85;
      }}
      .kc-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(133,72,255,0.12);
        border: 1px solid rgba(133,72,255,0.18);
      }}

      .kc-card {{
        background: #fff;
        border: 1px solid rgba(21,21,26,0.08);
        border-radius: var(--radius);
        padding: 14px 14px;
        box-shadow: 0 8px 24px rgba(21,21,26,0.06);
      }}
      .kc-card h3 {{
        font-size: 1.05rem;
        margin: 0 0 6px 0;
      }}
      .kc-note {{
        font-size: 0.9rem;
        opacity: 0.85;
        margin: 0;
      }}

      div.stButton > button {{
        border-radius: 12px !important;
        border: 0 !important;
        background: linear-gradient(90deg, var(--kc-deep), var(--kc-violet)) !important;
        color: white !important;
        padding: 0.6rem 0.9rem !important;
        font-weight: 700 !important;
      }}
      div.stButton > button:hover {{
        filter: brightness(1.03);
      }}
      div.stDownloadButton > button {{
        border-radius: 12px !important;
        border: 1px solid rgba(21,21,26,0.12) !important;
        background: white !important;
        color: var(--kc-text) !important;
        font-weight: 700 !important;
      }}
      div.stDownloadButton > button:hover {{
        background: rgba(133,72,255,0.06) !important;
        border-color: rgba(133,72,255,0.28) !important;
      }}

      .stTextInput input, .stDateInput input {{
        border-radius: 12px !important;
      }}

      button[data-baseweb="tab"] {{
        border-radius: 999px !important;
      }}

      div[data-testid="stDataFrame"] {{
        border-radius: var(--radius);
        overflow: hidden;
        border: 1px solid rgba(21,21,26,0.08);
      }}

      .kc-footer {{
        margin-top: 14px;
        opacity: 0.7;
        font-size: 0.85rem;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header():
    c1, c2 = st.columns([1, 6])
    with c1:
        try:
            st.image("assets/KrispCallLogo.png", use_container_width=True)
        except Exception:
            st.write("")
    with c2:
        st.markdown(
            f"""
            <div class="kc-header">
              <div>
                <div class="kc-title">{APP_TITLE}</div>
                <div class="kc-subtitle">{APP_SUBTITLE}</div>
              </div>
              <div style="margin-left:auto" class="kc-badge">KrispCall branded</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def _normalize_email(s: object):
    if s is None:
        return None
    s = str(s).strip().lower()
    if not s or s == "nan":
        return None
    return s


_EMAIL_RE = re.compile(r"([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})", re.IGNORECASE)


def _extract_email_from_text(txt: object):
    if txt is None:
        return None
    m = _EMAIL_RE.search(str(txt))
    return m.group(1).lower() if m else None


def _parse_unix_time_to_utc_dt(series: pd.Series) -> pd.Series:
    t = pd.to_numeric(series, errors="coerce")
    if t.notna().all():
        is_ms = float(t.median()) > 1e11
        if is_ms:
            t = (t // 1000)
        return pd.to_datetime(t, unit="s", utc=True, errors="coerce")
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt


def dedupe_mixpanel_export(df: pd.DataFrame) -> pd.DataFrame:
    required = ["event", "distinct_id", "time", "$insert_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    df = df.copy()

    t = pd.to_numeric(df["time"], errors="coerce")
    if t.notna().all():
        if float(t.median()) > 1e11:
            t = (t // 1000)
        df["_time_s"] = t.astype("Int64")
    else:
        dt = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df["_time_s"] = (dt.view("int64") // 10**9).astype("Int64")

    sort_cols = ["_time_s"]
    if "mp_processing_time_ms" in df.columns:
        sort_cols = ["mp_processing_time_ms"] + sort_cols

    df = df.sort_values(sort_cols, kind="mergesort")
    df = df.drop_duplicates(
        subset=["event", "distinct_id", "_time_s", "$insert_id"],
        keep="last"
    )
    df = df.drop(columns=["_time_s"])
    return df


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_mixpanel_npm(to_date: date) -> pd.DataFrame:
    mp = st.secrets["mixpanel"]
    project_id = str(mp["project_id"]).strip()
    from_date = str(mp["from_date"]).strip()
    to_date_str = to_date.isoformat()

    events = ["New Payment Made"]
    event_array_json = json.dumps(events)

    base_url = mp.get("base_url", "https://data-eu.mixpanel.com")
    url = (
        f"{base_url}/api/2.0/export"
        f"?project_id={project_id}"
        f"&from_date={from_date}"
        f"&to_date={to_date_str}"
        f"&event={event_array_json}"
    )

    headers = {
        "accept": "text/plain",
        "authorization": str(mp["auth_header"]).strip(),
    }

    rows = []
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Mixpanel export failed. Status {r.status_code}. Body: {r.text[:500]}")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    if "properties" not in raw.columns:
        return raw

    props = pd.json_normalize(raw["properties"])
    df = pd.concat([raw.drop(columns=["properties"]), props], axis=1)
    df = dedupe_mixpanel_export(df)
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols:
            return cols[key]
    return None


def summarize(deals_enriched: pd.DataFrame, connected_only: bool) -> pd.DataFrame:
    df = deals_enriched.copy()

    if connected_only:
        df = df[df["Connected"] == True].copy()

    df = df[df["DealMonth"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        total = len(g)
        churn = int(g["Churned (AsOf MonthEnd)"].fillna(True).sum())
        churn_pct = (churn / total) if total else np.nan
        upsell_sum = float(g["Upsell Net Change"].fillna(0).sum())
        upsell_pos_sum = float(g["Upsell Positive Only"].fillna(0).sum())
        annual_active = int(g["Annual Active (AsOf MonthEnd)"].fillna(False).sum())
        multi_curr = int(g["Current Month Renew Multiple Flag"].fillna(False).sum())
        multi_prev = int(g["Previous Month Renew Multiple Flag"].fillna(False).sum())
        return pd.Series({
            "Accounts": total,
            "Churned": churn,
            "Churn %": churn_pct,
            "Annual Active": annual_active,
            "Upsell Net Change Sum": upsell_sum,
            "Upsell Positive Only Sum": upsell_pos_sum,
            "Rows with Multi Current Month Txns": multi_curr,
            "Rows with Multi Previous Month Txns": multi_prev,
        })

    overall = df.groupby("DealMonth", as_index=False).apply(_agg).reset_index(drop=True)
    overall["Scope"] = "Overall"
    overall["Deal Owner"] = "All"

    if "Deal - Owner" in df.columns:
        by_owner = df.groupby(["DealMonth", "Deal - Owner"], as_index=False).apply(_agg).reset_index(drop=True)
        by_owner = by_owner.rename(columns={"Deal - Owner": "Deal Owner"})
        by_owner["Scope"] = "By Owner"
    else:
        by_owner = pd.DataFrame()

    out = pd.concat([overall, by_owner], ignore_index=True)
    out = out.sort_values(["DealMonth", "Scope", "Deal Owner"], kind="mergesort")
    return out


def build_enriched_deals(
    deals_df: pd.DataFrame,
    npm_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deal_date_col = "Deal - Deal created on"
    deal_email_col = "Person - Email"
    deal_owner_col = "Deal - Owner"
    deal_label_col = "Deal - Label"
    deal_value_col = "Deal - Deal value"

    missing_deals = [c for c in [deal_date_col, deal_email_col, deal_owner_col, deal_label_col] if c not in deals_df.columns]
    if missing_deals:
        raise KeyError(f"Deals file is missing required columns: {missing_deals}")

    deals = deals_df.copy()

    deals["_deal_created_dt"] = pd.to_datetime(deals[deal_date_col], errors="coerce", utc=True)
    if deals["_deal_created_dt"].isna().all():
        deals["_deal_created_dt"] = pd.to_datetime(deals[deal_date_col], errors="coerce")

    deals["DealMonth"] = deals["_deal_created_dt"].dt.to_period("M").dt.to_timestamp()
    deals["EmailKey"] = deals[deal_email_col].map(_normalize_email)

    def _tier(label: object):
        if label is None:
            return None
        s = str(label).lower()
        for k in ["bronze", "silver", "gold", "platinum", "vip"]:
            if k in s:
                return k.title()
        return None

    deals["Tier"] = deals[deal_label_col].map(_tier)

    # Connected is TRUE if label contains Connected but not Not Connected
    def _connected(label: object) -> bool:
        if label is None:
            return False
        s = str(label).lower()
        return ("connected" in s) and ("not connected" not in s)

    deals["Connected"] = deals[deal_label_col].map(_connected)

    deals["_is_pipedrive_krispcall"] = (deals[deal_owner_col].astype(str).str.strip().str.lower() == "pipedrive krispcall").astype(int)

    if deal_value_col in deals.columns:
        deals["_deal_value_num"] = pd.to_numeric(deals[deal_value_col], errors="coerce").fillna(0.0)
    else:
        deals["_deal_value_num"] = 0.0

    deals["_dedup_key"] = deals["EmailKey"].fillna("__missing_email__") + "|" + deals["DealMonth"].astype(str)

    grp_counts = deals.groupby("_dedup_key")["_dedup_key"].transform("count")
    deals["Dedup Group Count"] = grp_counts
    deals["Dedup Dropped Duplicates"] = grp_counts.gt(1)

    deals_sorted = deals.sort_values(
        by=["_dedup_key", "_is_pipedrive_krispcall", "_deal_value_num", "_deal_created_dt"],
        ascending=[True, True, False, False],
        kind="mergesort"
    )
    deals_dedup = deals_sorted.drop_duplicates(subset=["_dedup_key"], keep="first").copy()

    if npm_df is None or npm_df.empty:
        deals_out = deals_dedup.drop(columns=["_dedup_key"], errors="ignore").copy()
        for c in [
            "Current Month Renew Amount","Previous Month Renew Amount","Annual Payment Type (AsOf MonthEnd)",
            "Subscription Valid Till (AsOf MonthEnd)","Active Subscription (AsOf MonthEnd)","Churned (AsOf MonthEnd)",
            "Upsell Net Change","Upsell Positive Only"
        ]:
            if c not in deals_out.columns:
                deals_out[c] = np.nan
        summary_all = summarize(deals_out, connected_only=False)
        summary_connected = summarize(deals_out, connected_only=True)
        return deals_out, summary_all, summary_connected

    npm = npm_df.copy()

    col_email = _find_col(npm, ["$email", "$Email", "email"])
    col_amount_desc = _find_col(npm, ["Amount Description", "amount description", "amount_description", "Amount description"])
    col_amount = _find_col(npm, ["Amount", "amount"])
    col_breakdown = _find_col(npm, ["Amount breakdown", "amount breakdown", "amount_breakdown", "Amount Breakdown"])
    col_time = _find_col(npm, ["time"])

    if col_time is None or col_amount is None:
        raise KeyError("NPM export is missing required fields. Needed: time and Amount (case-insensitive match).")

    npm["PayDT"] = _parse_unix_time_to_utc_dt(npm[col_time])
    npm["PayMonth"] = npm["PayDT"].dt.to_period("M").dt.to_timestamp()
    npm["AmountNum"] = pd.to_numeric(npm[col_amount], errors="coerce")

    if col_email:
        npm["EmailKey"] = npm[col_email].map(_normalize_email)
    else:
        npm["EmailKey"] = None

    if col_amount_desc:
        parsed = npm[col_amount_desc].map(_extract_email_from_text)
        npm["EmailKey"] = npm["EmailKey"].fillna(parsed)

    npm_valid = npm[npm["EmailKey"].notna()].copy()

    desc = npm_valid[col_amount_desc].astype(str) if col_amount_desc else pd.Series([""] * len(npm_valid), index=npm_valid.index)
    breakdown = npm_valid[col_breakdown].astype(str) if col_breakdown else pd.Series([""] * len(npm_valid), index=npm_valid.index)

    annual_amount_threshold = 40
    start_mask = npm_valid["AmountNum"].fillna(0).gt(annual_amount_threshold) & desc.str.contains("workspace subscription", case=False, na=False)

    allowed_nums = ["9360", "14400", "11520", "12960", "10080", "10800", "38400", "34560", "30720", "26880"]
    breakdown_mask = breakdown.apply(lambda x: any(n in str(x) for n in allowed_nums))

    desc_no_comma = ~desc.str.contains(",", na=False)
    contains_email_exact = []
    if col_amount_desc:
        for _, row in npm_valid.iterrows():
            ek = row["EmailKey"]
            d = str(row[col_amount_desc]) if col_amount_desc else ""
            contains_email_exact.append(bool(ek) and ek in d.lower())
    else:
        contains_email_exact = [False] * len(npm_valid)
    contains_email_exact = pd.Series(contains_email_exact, index=npm_valid.index)

    renew_mask = npm_valid["AmountNum"].fillna(0).gt(annual_amount_threshold) & (
        (contains_email_exact & desc_no_comma) | breakdown_mask
    )

    annual_candidates = npm_valid[start_mask | renew_mask].copy()
    if not annual_candidates.empty:
        annual_candidates["Annual Payment Type"] = np.where(start_mask.loc[annual_candidates.index], "Subscription", "Renew")
    else:
        annual_candidates["Annual Payment Type"] = pd.Series(dtype="object")

    annual_user_set = set(annual_candidates["EmailKey"].unique())

    def _contains(s: pd.Series, pat: str) -> pd.Series:
        return s.str.contains(pat, case=False, na=False)

    d = desc

    cond_email_in_desc = []
    if col_amount_desc:
        for _, row in npm_valid.iterrows():
            ek = row["EmailKey"]
            txt = str(row[col_amount_desc]) if col_amount_desc else ""
            cond_email_in_desc.append(bool(ek) and ek in txt.lower())
    else:
        cond_email_in_desc = [False] * len(npm_valid)
    cond_email_in_desc = pd.Series(cond_email_in_desc, index=npm_valid.index)

    cond_number_purchased = _contains(d, "number purchased")
    cond_agent_added = _contains(d, "agent added") & d.str.contains(",", na=False)
    cond_number_renew = _contains(d, "number renew") & npm_valid["EmailKey"].isin(annual_user_set)
    cond_workspace_sub = _contains(d, "workspace subscription")

    renewal_mask = cond_email_in_desc | cond_number_purchased | cond_agent_added | cond_workspace_sub | cond_number_renew
    renewals = npm_valid[renewal_mask].copy()

    renewals = renewals[renewals["PayDT"].notna()].copy()
    renewals = renewals.sort_values(["EmailKey", "PayMonth", "PayDT"], kind="mergesort")

    grp = renewals.groupby(["EmailKey", "PayMonth"], as_index=False)
    txn_count = grp.size().rename(columns={"size": "Renew Txn Count"})
    idx_latest = grp["PayDT"].idxmax()

    latest_rows = renewals.loc[idx_latest].copy()
    latest_rows = latest_rows.merge(txn_count, on=["EmailKey", "PayMonth"], how="left")
    latest_rows["Renew Multiple Flag"] = latest_rows["Renew Txn Count"].fillna(0).astype(int) > 1

    latest_map = latest_rows.set_index(["EmailKey", "PayMonth"])

    deals_dedup["PrevDealMonth"] = (deals_dedup["DealMonth"] - pd.offsets.MonthBegin(1)).dt.to_period("M").dt.to_timestamp()

    def _lookup_amount(email, month):
        if email is None or pd.isna(month):
            return np.nan
        try:
            v = latest_map.loc[(email, month), "AmountNum"]
            return float(v) if pd.notna(v) else np.nan
        except KeyError:
            return np.nan

    def _lookup_date(email, month):
        if email is None or pd.isna(month):
            return pd.NaT
        try:
            return latest_map.loc[(email, month), "PayDT"]
        except KeyError:
            return pd.NaT

    def _lookup_cnt(email, month):
        if email is None or pd.isna(month):
            return 0
        try:
            return int(latest_map.loc[(email, month), "Renew Txn Count"])
        except KeyError:
            return 0

    def _lookup_mult(email, month):
        if email is None or pd.isna(month):
            return False
        try:
            return bool(latest_map.loc[(email, month), "Renew Multiple Flag"])
        except KeyError:
            return False

    deals_dedup["Current Month Renew Amount"] = [_lookup_amount(e, m) for e, m in zip(deals_dedup["EmailKey"], deals_dedup["DealMonth"])]
    deals_dedup["Current Month Renew Date"] = [_lookup_date(e, m) for e, m in zip(deals_dedup["EmailKey"], deals_dedup["DealMonth"])]
    deals_dedup["Current Month Renew Txn Count"] = [_lookup_cnt(e, m) for e, m in zip(deals_dedup["EmailKey"], deals_dedup["DealMonth"])]
    deals_dedup["Current Month Renew Multiple Flag"] = [_lookup_mult(e, m) for e, m in zip(deals_dedup["EmailKey"], deals_dedup["DealMonth"])]

    deals_dedup["Previous Month Renew Amount"] = [_lookup_amount(e, m) for e, m in zip(deals_dedup["EmailKey"], deals_dedup["PrevDealMonth"])]
    deals_dedup["Previous Month Renew Date"] = [_lookup_date(e, m) for e, m in zip(deals_dedup["EmailKey"], deals_dedup["PrevDealMonth"])]
    deals_dedup["Previous Month Renew Txn Count"] = [_lookup_cnt(e, m) for e, m in zip(deals_dedup["EmailKey"], deals_dedup["PrevDealMonth"])]
    deals_dedup["Previous Month Renew Multiple Flag"] = [_lookup_mult(e, m) for e, m in zip(deals_dedup["EmailKey"], deals_dedup["PrevDealMonth"])]

    deals_dedup["MonthEndDT"] = (deals_dedup["DealMonth"] + pd.offsets.MonthEnd(0)) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    deals_dedup["NextMonthStart"] = (deals_dedup["DealMonth"] + pd.offsets.MonthBegin(1)).dt.normalize()

    ren_simple = renewals[["EmailKey", "PayDT"]].dropna().sort_values(["EmailKey", "PayDT"])
    deals_cut = deals_dedup[["EmailKey", "MonthEndDT"]].dropna(subset=["EmailKey"]).sort_values(["EmailKey", "MonthEndDT"])

    merged_monthly = pd.merge_asof(
        deals_cut,
        ren_simple,
        left_on="MonthEndDT",
        right_on="PayDT",
        by="EmailKey",
        direction="backward",
        allow_exact_matches=True
    ).rename(columns={"PayDT": "Latest Monthly PayDT (AsOf MonthEnd)"})

    deals_dedup = deals_dedup.merge(merged_monthly, on=["EmailKey", "MonthEndDT"], how="left")

    def _add_month(dt):
        if pd.isna(dt):
            return pd.NaT
        if relativedelta:
            return (pd.Timestamp(dt).to_pydatetime().replace(tzinfo=None) + relativedelta(months=1))
        return pd.Timestamp(dt) + pd.DateOffset(months=1)

    def _add_year(dt):
        if pd.isna(dt):
            return pd.NaT
        if relativedelta:
            return (pd.Timestamp(dt).to_pydatetime().replace(tzinfo=None) + relativedelta(years=1))
        return pd.Timestamp(dt) + pd.DateOffset(years=1)

    deals_dedup["Monthly Valid Till (AsOf MonthEnd)"] = deals_dedup["Latest Monthly PayDT (AsOf MonthEnd)"].apply(_add_month)

    annual_simple = annual_candidates[["EmailKey", "PayDT", "Annual Payment Type"]].dropna().sort_values(["EmailKey", "PayDT"])
    deals_cut2 = deals_dedup[["EmailKey", "MonthEndDT"]].dropna(subset=["EmailKey"]).sort_values(["EmailKey", "MonthEndDT"])

    merged_annual = pd.merge_asof(
        deals_cut2,
        annual_simple,
        left_on="MonthEndDT",
        right_on="PayDT",
        by="EmailKey",
        direction="backward",
        allow_exact_matches=True
    ).rename(columns={"PayDT": "Latest Annual PayDT (AsOf MonthEnd)", "Annual Payment Type": "Annual Payment Type (AsOf MonthEnd)"})

    deals_dedup = deals_dedup.merge(merged_annual, on=["EmailKey", "MonthEndDT"], how="left")
    deals_dedup["Annual Valid Till (AsOf MonthEnd)"] = deals_dedup["Latest Annual PayDT (AsOf MonthEnd)"].apply(_add_year)

    deals_dedup["Subscription Valid Till (AsOf MonthEnd)"] = deals_dedup[["Monthly Valid Till (AsOf MonthEnd)", "Annual Valid Till (AsOf MonthEnd)"]].max(axis=1)

    deals_dedup["Annual Active (AsOf MonthEnd)"] = deals_dedup["Annual Valid Till (AsOf MonthEnd)"].notna() & (
        deals_dedup["Annual Valid Till (AsOf MonthEnd)"] >= deals_dedup["NextMonthStart"]
    )
    deals_dedup["Active Subscription (AsOf MonthEnd)"] = deals_dedup["Subscription Valid Till (AsOf MonthEnd)"].notna() & (
        deals_dedup["Subscription Valid Till (AsOf MonthEnd)"] >= deals_dedup["NextMonthStart"]
    )
    deals_dedup["Churned (AsOf MonthEnd)"] = ~deals_dedup["Active Subscription (AsOf MonthEnd)"]

    prev_amt = deals_dedup["Previous Month Renew Amount"].fillna(0.0)
    curr_amt = deals_dedup["Current Month Renew Amount"].fillna(0.0)
    eligible = (prev_amt > 0) & (~deals_dedup["Churned (AsOf MonthEnd)"]) & (~deals_dedup["Annual Active (AsOf MonthEnd)"])
    deals_dedup["Upsell Net Change"] = np.where(eligible, (curr_amt - prev_amt), 0.0)
    deals_dedup["Upsell Positive Only"] = np.where(deals_dedup["Upsell Net Change"] > 0, deals_dedup["Upsell Net Change"], 0.0)

    deals_out = deals_dedup.drop(columns=["_dedup_key"], errors="ignore").copy()
    summary_all = summarize(deals_out, connected_only=False)
    summary_connected = summarize(deals_out, connected_only=True)
    return deals_out, summary_all, summary_connected


def make_excel(deals_raw: pd.DataFrame, deals_enriched: pd.DataFrame, summary_all: pd.DataFrame, summary_connected: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        deals_enriched.to_excel(writer, sheet_name="Deals_enriched", index=False)
        summary_all.to_excel(writer, sheet_name="Summary_all", index=False)
        summary_connected.to_excel(writer, sheet_name="Summary_connected", index=False)
        deals_raw.to_excel(writer, sheet_name="Deals_raw", index=False)

        audit_rows = [
            ("Deals rows raw", len(deals_raw)),
            ("Deals rows enriched (deduped)", len(deals_enriched)),
            ("Deals connected TRUE", int((deals_enriched["Connected"] == True).sum())),
            ("Deals churned (AsOf MonthEnd)", int(deals_enriched["Churned (AsOf MonthEnd)"].fillna(True).sum())),
            ("Deals annual active (AsOf MonthEnd)", int(deals_enriched["Annual Active (AsOf MonthEnd)"].fillna(False).sum())),
        ]
        audit = pd.DataFrame(audit_rows, columns=["Metric", "Value"])
        audit.to_excel(writer, sheet_name="Audit", index=False)

    return output.getvalue()


def login_gate():
    if st.session_state.get("authenticated"):
        return True

    st.markdown('<div class="kc-card"><h3>Login</h3><p class="kc-note">Enter credentials to access the dashboard.</p></div>', unsafe_allow_html=True)
    u = st.text_input("Username", value="", key="login_user")
    p = st.text_input("Password", value="", type="password", key="login_pass")

    if st.button("Sign in", type="primary"):
        if u == str(st.secrets["auth"]["username"]) and p == str(st.secrets["auth"]["password"]):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False


def kpi_row(summary_df: pd.DataFrame):
    if summary_df is None or summary_df.empty:
        st.info("No summary available for the selected scope.")
        return

    overall = summary_df[summary_df["Scope"] == "Overall"].copy()
    if overall.empty:
        st.info("No overall summary available.")
        return

    latest = overall.sort_values("DealMonth").iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accounts", int(latest["Accounts"]))
    c2.metric("Churned", int(latest["Churned"]))
    churn_pct = float(latest["Churn %"]) * 100 if pd.notna(latest["Churn %"]) else np.nan
    c3.metric("Churn %", f"{churn_pct:.2f}%" if pd.notna(churn_pct) else "NA")
    c4.metric("Annual Active", int(latest["Annual Active"]))
    c5.metric("Upsell Sum", f'{float(latest["Upsell Net Change Sum"]):,.2f}')


def main():
    st.set_page_config(page_title="KrispCall. Account Management", page_icon="📞", layout="wide")
    _require_secrets()
    inject_brand_css()
    render_header()

    st.sidebar.image("assets/KrispCallLogo.png", use_container_width=True)
    st.sidebar.markdown("### Controls")

    if not login_gate():
        return

    end_date = st.sidebar.date_input("Payments to date", value=date.today())
    st.sidebar.caption("Payments are fetched from Mixpanel Export API. Start date is fixed to 2023-05-01.")
    deals_file = st.sidebar.file_uploader("Upload Deals CSV", type=["csv"])

    fetch_btn = st.sidebar.button("Fetch payments", type="primary")
    focus_connected = st.sidebar.toggle("Focus on Connected only", value=False)

    if deals_file is None:
        st.info("Upload your Deals CSV to begin.")
        return

    deals_raw = pd.read_csv(deals_file)

    if "npm_cached" not in st.session_state:
        st.session_state["npm_cached"] = None

    if fetch_btn:
        with st.spinner("Fetching New Payment Made events from Mixpanel..."):
            try:
                st.session_state["npm_cached"] = fetch_mixpanel_npm(end_date)
                st.success("Payments fetched.")
            except Exception as e:
                st.error(str(e))

    npm_df = st.session_state.get("npm_cached")
    if npm_df is None:
        st.warning("Click Fetch payments to load Mixpanel events for calculations.")
        return

    with st.spinner("Building enriched dataset..."):
        deals_enriched, summary_all, summary_connected = build_enriched_deals(deals_raw, npm_df)

    summary_view = summary_connected if focus_connected else summary_all

    kpi_row(summary_view)

    tab1, tab2, tab3, tab4 = st.tabs(["Summary tables", "Visuals", "Deals enriched", "Payments preview"])

    with tab1:
        st.markdown('<div class="kc-card"><h3>Summary</h3><p class="kc-note">Monthwise churn and upsell. Computed as of each month end.</p></div>', unsafe_allow_html=True)
        st.write("")
        st.subheader("All deals")
        st.dataframe(summary_all, use_container_width=True)
        st.subheader("Connected only")
        st.dataframe(summary_connected, use_container_width=True)

    with tab2:
        st.markdown('<div class="kc-card"><h3>Trends</h3><p class="kc-note">Charts follow your selected focus in the sidebar.</p></div>', unsafe_allow_html=True)
        st.write("")
        src = summary_view
        overall = src[src["Scope"] == "Overall"].copy() if not src.empty else pd.DataFrame()
        if overall.empty:
            st.info("No data to chart.")
        else:
            overall["DealMonth"] = pd.to_datetime(overall["DealMonth"])
            overall = overall.sort_values("DealMonth")
            st.caption("Churn percentage over time")
            st.line_chart(overall.set_index("DealMonth")[["Churn %"]])

            st.caption("Churned accounts over time")
            st.line_chart(overall.set_index("DealMonth")[["Churned"]])

            st.caption("Upsell net change sum over time")
            st.line_chart(overall.set_index("DealMonth")[["Upsell Net Change Sum"]])

    with tab3:
        st.markdown('<div class="kc-card"><h3>Deals dataset</h3><p class="kc-note">Deduplicated per email-month. Includes Connected flag, payments, validity, churn, upsell.</p></div>', unsafe_allow_html=True)
        st.write("")
        st.dataframe(deals_enriched, use_container_width=True)

    with tab4:
        st.markdown('<div class="kc-card"><h3>Payments dataset</h3><p class="kc-note">Mixpanel export rows after deduplication. This is what drives renew mapping.</p></div>', unsafe_allow_html=True)
        st.write("")
        st.dataframe(npm_df.head(200), use_container_width=True)

    st.divider()
    st.subheader("Export")
    excel_bytes = make_excel(deals_raw, deals_enriched, summary_all, summary_connected)
    st.download_button(
        "Download Excel workbook",
        data=excel_bytes,
        file_name="account_mgmt_upsell_churn_enriched.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown(
        '<div class="kc-footer">Built for KrispCall. Data is computed using month-end cutoffs for churn and validity.</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
