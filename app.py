import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Floorball Value Betting", page_icon="🥍", layout="wide")
st.title("🥍 Floorball Value Betting App")
st.write("Поиск валуйных ставок на основе математической модели (с регуляризацией).")

@st.cache_data
def load_data():
    return pd.read_csv('floorball_dataset.csv')

df = load_data()
matches_with_score = df.dropna(subset=['home_score', 'away_score']).copy()

avg_home_goals = matches_with_score['home_score'].mean()
avg_away_goals = matches_with_score['away_score'].mean()

home_stats = matches_with_score.groupby('home_team').agg({'home_score': 'mean', 'away_score': 'mean'})
away_stats = matches_with_score.groupby('away_team').agg({'away_score': 'mean', 'home_score': 'mean'})

home_attack = home_stats['home_score']
home_defense = home_stats['away_score']
away_attack = away_stats['away_score']
away_defense = away_stats['home_score']

def predict_probabilities(home, away):
    h_att = home_attack.get(home, avg_home_goals)
    h_def = home_defense.get(home, avg_away_goals)
    a_att = away_attack.get(away, avg_away_goals)
    a_def = away_defense.get(away, avg_home_goals)
    
    # Считаем ожидаемые голы
    exp_home = (h_att / avg_home_goals) * (a_def / avg_away_goals) * avg_home_goals
    exp_away = (a_att / avg_away_goals) * (h_def / avg_home_goals) * avg_away_goals
    
    # СГЛАЖИВАНИЕ (Shrinkage): притягиваем ожидаемые голы к среднему по лиге
    # Это не даст модели выдавать нереальные вероятности
    exp_home = exp_home * 0.6 + avg_home_goals * 0.4
    exp_away = exp_away * 0.6 + avg_away_goals * 0.4
    
    p_home, p_draw, p_away = 0, 0, 0
    for h_g in range(15):
        for a_g in range(15):
            p = poisson.pmf(h_g, exp_home) * poisson.pmf(a_g, exp_away)
            if h_g > a_g: p_home += p
            elif h_g == a_g: p_draw += p
            else: p_away += p
            
    # Ограничиваем максимальную вероятность (максимум 85%)
    return min(p_home, 0.85), p_draw, min(p_away, 0.85)

# Берем будущие матчи
future_matches = df[df['home_score'].isna()].copy()

if len(future_matches) > 0:
    st.subheader("📊 Предстоящие матчи и прогнозы")
    
    # Считаем вероятности для будущих матчей
    future_matches[['p_home', 'p_draw', 'p_away']] = future_matches.apply(
        lambda row: pd.Series(predict_probabilities(row['home_team'], row['away_team'])), axis=1
    )
    
    for index, row in future_matches.iterrows():
        probs = {'П1': row['p_home'], 'Х': row['p_draw'], 'П2': row['p_away']}
        best_outcome = max(probs, key=probs.get)
        best_prob = probs[best_outcome]
        fair_odds = 1 / best_prob
        
        # ФИЛЬТР: Показываем только те матчи, где есть смысл ставить (кэф > 1.30)
        if fair_odds > 1.30:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"{row['home_team']} (Дом)")
            with col2:
                st.write("vs")
            with col3:
                st.write(f"{row['away_team']} (Гости)")
            
            if best_prob > 0.60:
                st.success(f"🔥 Рекомендация: {best_outcome} | Вероятность: {best_prob*100:.1f}% | Мин. кэф: {fair_odds:.2f}")
            else:
                st.warning(f"⚖️ Возможный валуй: {best_outcome} | Вероятность: {best_prob*100:.1f}% | Мин. кэф: {fair_odds:.2f}")
                
            st.divider()
else:
    st.warning("Нет предстоящих матчей для анализа.")
    # ==========================================
# КАЛЬКУЛЯТОР VALUE (EV)
# ==========================================
st.divider()
st.header("🧮 Калькулятор Value (EV)")
st.write("Введите коэффициент из вашего букмекера, и приложение покажет, есть ли в ставке математическое преимущество.")

col_team1, col_team2 = st.columns(2)
home_input = col_team1.text_input("Команда хозяев")
away_input = col_team2.text_input("Команда гостей")

if home_input and away_input:
    # Считаем вероятности через нашу модель
    p_home, p_draw, p_away = predict_probabilities(home_input, away_input)
    
    st.write(f"Вероятность модели: П1 ({p_home*100:.1f}%) | Х ({p_draw*100:.1f}%) | П2 ({p_away*100:.1f}%)")
    
    col_odds1, col_odds2, col_odds3 = st.columns(3)
    odds_home = col_odds1.number_input("Кэф на П1", min_value=1.01, value=1.80, step=0.05)
    odds_draw = col_odds2.number_input("Кэф на Х", min_value=1.01, value=6.00, step=0.10)
    odds_away = col_odds3.number_input("Кэф на П2", min_value=1.01, value=4.00, step=0.05)
    
    # Считаем EV для каждого исхода (EV = (Вероятность * Кэф) - 1)
    ev_home = (p_home * odds_home) - 1
    ev_draw = (p_draw * odds_draw) - 1
    ev_away = (p_away * odds_away) - 1
    
    st.write("---")
    col_ev1, col_ev2, col_ev3 = st.columns(3)
    
    # Вывод П1
    with col_ev1:
        if ev_home > 0:
            st.success(f"П1: EV = +{ev_home*100:.1f}%\n\n🔥 ВАЛУЙ! Ставить.")
        else:
            st.error(f"П1: EV = {ev_home*100:.1f}%\n\n❌ Не ставить.")
            
    # Вывод Ничьей
    with col_ev2:
        if ev_draw > 0:
            st.success(f"Х: EV = +{ev_draw*100:.1f}%\n\n🔥 ВАЛУЙ! Ставить.")
        else:
            st.error(f"Х: EV = {ev_draw*100:.1f}%\n\n❌ Не ставить.")
            
    # Вывод П2
    with col_ev3:
        if ev_away > 0:
            st.success(f"П2: EV = +{ev_away*100:.1f}%\n\n🔥 ВАЛУЙ! Ставить.")
        else:
            st.error(f"П2: EV = {ev_away*100:.1f}%\n\n❌ Не ставить.")
