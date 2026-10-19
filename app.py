import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Floorball Value Terminal", page_icon="🏑", layout="wide")
st.title("🏑 Floorball Value Terminal")
st.markdown("Профессиональная аналитика, поиск валуя и банкролл-менеджмент.")

@st.cache_data
def load_data():
    return pd.read_csv('floorball_dataset.csv')

df = load_data()
matches_with_score = df.dropna(subset=['home_score', 'away_score']).copy()

avg_home_goals = matches_with_score['home_score'].mean()
avg_away_goals = matches_with_score['away_score'].mean()

home_stats = matches_with_score.groupby('home_team').agg({'home_score': 'mean', 'away_score': 'mean'})
away_stats = matches_with_score.groupby('away_team').agg({'away_score': 'mean', 'home_score': 'mean'})

def get_team_stats(team_name, is_home):
    if is_home:
        att = home_stats['home_score'].get(team_name, avg_home_goals)
        defs = home_stats['away_score'].get(team_name, avg_away_goals)
    else:
        att = away_stats['away_score'].get(team_name, avg_away_goals)
        defs = away_stats['home_score'].get(team_name, avg_home_goals)
    return att, defs

def get_recent_form(team_name, is_home):
    team_matches = df[((df['home_team'] == team_name) & df['home_score'].notna()) | 
                      ((df['away_team'] == team_name) & df['away_score'].notna())]
    recent = team_matches.tail(5)
    form = []
    for _, row in recent.iterrows():
        if row['home_team'] == team_name:
            if row['home_score'] > row['away_score']: form.append("🟢")
            elif row['home_score'] < row['away_score']: form.append("🔴")
            else: form.append("🟡")
        else:
            if row['away_score'] > row['home_score']: form.append("🟢")
            elif row['away_score'] < row['home_score']: form.append("🔴")
            else: form.append("🟡")
    return "".join(form) if form else "Нет данных"

def predict_probabilities(home, away):
    h_att, h_def = get_team_stats(home, is_home=True)
    a_att, a_def = get_team_stats(away, is_home=False)
    
    exp_home = (h_att / avg_home_goals) * (a_def / avg_away_goals) * avg_home_goals
    exp_away = (a_att / avg_away_goals) * (h_def / avg_home_goals) * avg_away_goals
    
    # Защита от аномалий и сглаживание
    exp_home = min(exp_home, 5.5)
    exp_away = min(exp_away, 4.5)
    exp_home = exp_home * 0.5 + avg_home_goals * 0.5
    exp_away = exp_away * 0.5 + avg_away_goals * 0.5
    
    p_home, p_draw, p_away = 0, 0, 0
    total_over_95, total_over_105 = 0, 0
    handicap_home_minus1 = 0
    
    for h_g in range(20):
        for a_g in range(20):
            p = poisson.pmf(h_g, exp_home) * poisson.pmf(a_g, exp_away)
            if h_g > a_g: p_home += p
            elif h_g == a_g: p_draw += p
            else: p_away += p
            if h_g + a_g > 9.5: total_over_95 += p
            if h_g + a_g > 10.5: total_over_105 += p
            if (h_g - a_g) >= 2: handicap_home_minus1 += p
            
    return {
        'exp_home': exp_home, 'exp_away': exp_away,
        'p_home': min(p_home, 0.75), 'p_draw': p_draw, 'p_away': min(p_away, 0.75),
        'over_9_5': total_over_95, 'over_10_5': total_over_105,
        'handicap_home_minus1': handicap_home_minus1
    }

# --- СОЗДАЕМ ВКЛАДКИ ---
tab1, tab2, tab3 = st.tabs(["📊 Матчи и Прогнозы", "🧮 Калькулятор Value (Келли)", "📈 Тоталы и Форы"])

# --- ВКЛАДКА 1: МАТЧИ ---
with tab1:
    future_matches = df[df['home_score'].isna()].copy()
    
    if len(future_matches) > 0:
        for index, row in future_matches.iterrows():
            probs = predict_probabilities(row['home_team'], row['away_team'])
            outcomes = {'П1': probs['p_home'], 'Х': probs['p_draw'], 'П2': probs['p_away']}
            best_outcome = max(outcomes, key=outcomes.get)
            best_prob = outcomes[best_outcome]
            fair_odds = 1 / best_prob
            
            if fair_odds > 1.25:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.markdown(f"### {row['home_team']}")
                        st.markdown(f"Форма: **{get_recent_form(row['home_team'], is_home=True)}**")
                        st.metric(label="xG (Ожидаемые голы)", value=f"{probs['exp_home']:.2f}")
                    with col2:
                        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='text-align: center;'>{probs['exp_home']:.2f} : {probs['exp_away']:.2f}</h4>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"### {row['away_team']}")
                        st.markdown(f"Форма: **{get_recent_form(row['away_team'], is_home=False)}**")
                        st.metric(label="xG (Ожидаемые голы)", value=f"{probs['exp_away']:.2f}")
                    
                    st.markdown(f"**Вероятности:** П1: `{probs['p_home']*100:.1f}%` | Х: `{probs['p_draw']*100:.1f}%` | П2: `{probs['p_away']*100:.1f}%`")
                    if best_prob > 0.60:
                        st.success(f"🔥 VALUE FOUND: **{best_outcome}** | Вероятность: {best_prob*100:.1f}% | Честный кэф: {fair_odds:.2f}")
                    elif best_prob > 0.45:
                        st.warning(f"⚖️ ВОЗМОЖНЫЙ ВАЛУЙ: **{best_outcome}** | Вероятность: {best_prob*100:.1f}% | Честный кэф: {fair_odds:.2f}")

# --- ВКЛАДКА 2: КАЛЬКУЛЯТОР ---
with tab2:
    col_home, col_away = st.columns(2)
    home_input = col_home.text_input("Команда хозяев (например, TPS)")
    away_input = col_away.text_input("Команда гостей (например, Pss)")
    bank = st.number_input("Ваш банк (в рублях/BYN):", min_value=100, value=10000, step=100)

    if home_input and away_input:
        probs = predict_probabilities(home_input, away_input)
        st.write(f"Ожидаемые голы (λ): **{probs['exp_home']:.2f} : {probs['exp_away']:.2f}**")
        
        col_odds1, col_odds2, col_odds3 = st.columns(3)
        odds_home = col_odds1.number_input("Кэф на П1", min_value=1.01, value=1.80, step=0.05)
        odds_draw = col_odds2.number_input("Кэф на Х", min_value=1.01, value=6.00, step=0.10)
        odds_away = col_odds3.number_input("Кэф на П2", min_value=1.01, value=4.00, step=0.05)
        
        ev_home = (probs['p_home'] * odds_home) - 1
        kelly_home = max(0, ((probs['p_home'] * odds_home) - 1) / (odds_home - 1)) * 0.25
        ev_away = (probs['p_away'] * odds_away) - 1
        kelly_away = max(0, ((probs['p_away'] * odds_away) - 1) / (odds_away - 1)) * 0.25
        
        col_ev1, col_ev2, col_ev3 = st.columns(3)
        with col_ev1:
            if ev_home > 0:
                st.success(f"🔥 **П1 ВАЛУЙ!**\n\nEV: **+{ev_home*100:.1f}%**\n\nКелли: {kelly_home*100:.1f}%\n\nСумма: **{bank * kelly_home:.2f}**")
            else:
                st.error(f"❌ **П1 НЕ СТАВИТЬ**\n\nEV: {ev_home*100:.1f}%")
        with col_ev2:
            st.info(f"Х: EV {((probs['p_draw'] * odds_draw) - 1)*100:.1f}%")
        with col_ev3:
            if ev_away > 0:
                st.success(f"🔥 **П2 ВАЛУЙ!**\n\nEV: **+{ev_away*100:.1f}%**\n\nКелли: {kelly_away*100:.1f}%\n\nСумма: **{bank * kelly_away:.2f}**")
            else:
                st.error(f"❌ **П2 НЕ СТАВИТЬ**\n\nEV: {ev_away*100:.1f}%")

# --- ВКЛАДКА 3: ТОТАЛЫ И ФОРЫ ---
with tab3:
    st.subheader("Аналитика дополнительных рынков")
    st.write("Введите те же команды, что и во вкладке калькулятора, чтобы увидеть вероятности тоталов и фор.")
    
    # Дублируем ввод для удобства на 3-й вкладке (или используем значения из tab2)
    # Для простоты Streamlit пересчитает их, если они введены
    if 'home_input' in locals() and home_input and away_input:
        probs = predict_probabilities(home_input, away_input)
        col_tot1, col_tot2 = st.columns(2)
        with col_tot1:
            st.markdown(f"**Тоталы (Больше) для матча {home_input} - {away_input}**")
            st.write(f"ТБ 9.5: Вероятность **{probs['over_9_5']*100:.1f}%** (Честный кэф: {1/probs['over_9_5']:.2f})")
            st.write(f"ТБ 10.5: Вероятность **{probs['over_10_5']*100:.1f}%** (Честный кэф: {1/probs['over_10_5']:.2f})")
        with col_tot2:
            st.markdown("**Форы**")
            st.write(f"Фора хозяев (-1.0): Вероятность **{probs['handicap_home_minus1']*100:.1f}%** (Честный кэф: {1/probs['handicap_home_minus1']:.2f})")
            st.write("*Примечание: Фора -1.0 означает победу хозяев в 2 и более мяча.*")
    else:
        st.info("Пожалуйста, сначала введите названия команд во вкладке 'Калькулятор Value'.")
