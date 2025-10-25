# src/14_recommend_smart.py
import numpy as np
import joblib
import os
import itertools
from tabulate import tabulate

# === ПРАВИЛЬНАЯ проверка Streamlit ===
def is_streamlit():
    try:
        import streamlit as st
        return hasattr(st, 'runtime') and st.runtime.exists()
    except:
        return False

# === Пути ===
# Для локальной разработки и Streamlit Cloud
if os.path.exists('models/soil_predictor.pkl'):
    # Если app.py в корне (Streamlit Cloud)
    model_path = 'models/soil_predictor.pkl'
elif os.path.exists('../models/soil_predictor.pkl'):
    # Если app.py в папке src
    model_path = '../models/soil_predictor.pkl'
else:
    # Поиск от текущей директории
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'models', 'soil_predictor.pkl')

print(f"[DEBUG] Поиск модели по пути: {model_path}")
print(f"[DEBUG] Текущая директория: {os.getcwd()}")
print(f"[DEBUG] Файлы в текущей директории: {os.listdir('.')}")

if not os.path.exists(model_path):
    # Попытка найти модель в любом месте
    possible_paths = [
        'soil_predictor.pkl',
        'models/soil_predictor.pkl',
        '../models/soil_predictor.pkl',
        '../../models/soil_predictor.pkl'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"[DEBUG] Модель найдена: {model_path}")
            break
    else:
        raise FileNotFoundError(
            f"Модель не найдена!\n"
            f"Текущая директория: {os.getcwd()}\n"
            f"Искали в: {possible_paths}\n"
            f"Доступные файлы: {os.listdir('.')}"
        )

model = joblib.load(model_path)
print(f"[DEBUG] Модель успешно загружена из {model_path}")

# === Параметры ===
PARAMS = ['Сорг.%', 'Пористость', 'Мин. N']
action_names = ['Растения', 'Загрязнение', 'Биочар', 'Нитрификаторы', 'ПАУ-деструкторы']
combos = list(itertools.product([0, 1], repeat=5))

# === Предсказание ===
def predict(combo):
    X = np.array([combo])
    pred = model.predict(X)[0]
    return {
        'Сорг.%': round(pred[0], 2),
        'Пористость': round(pred[1], 2),
        'Мин. N': round(pred[2], 1)
    }

# === Все варианты ===
all_variants = [(combo, predict(combo)) for combo in combos]

# === Умный диапазон ===
def get_range(values):
    if not values:
        return "—"
    vals = sorted(set(values))
    if len(vals) == 1:
        return f"{vals[0]:.1f}".rstrip('0').rstrip('.') if vals[0] % 1 else f"{int(vals[0])}"
    
    intervals = []
    start = prev = vals[0]
    for v in vals[1:]:
        if abs(v - prev) > 0.05:
            intervals.append(format_interval(start, prev))
            start = v
        prev = v
    intervals.append(format_interval(start, prev))
    return "; ".join(intervals)

def format_interval(start, end):
    if abs(start - end) < 0.01:
        return f"{start:.1f}".rstrip('0').rstrip('.') if start % 1 else f"{int(start)}"
    else:
        s = f"{start:.1f}".rstrip('0').rstrip('.') if start % 1 else f"{int(start)}"
        e = f"{end:.1f}".rstrip('0').rstrip('.') if end % 1 else f"{int(end)}"
        return f"{s}–{e}"

# === Фильтрация ===
def filter_variants(variants, param, target, tolerance=0.15):
    filtered = [v for v in variants if abs(v[1][param] - target) <= tolerance * max(1, abs(target))]
    print(f"[DEBUG] Фильтрация {param}={target}: {len(variants)} → {len(filtered)} вариантов")
    return filtered

# === Топ-3 ===
def get_top3(variants, targets):
    return sorted(
        [(sum(abs(pred[p] - targets.get(p, pred[p])) for p in PARAMS), combo, pred)
         for combo, pred in variants]
    )[:3]

# === Получить диапазоны параметров ===
def get_param_ranges(variants):
    ranges = {}
    for param in PARAMS:
        vals = [v[1][param] for v in variants]
        if vals:
            ranges[param] = {
                'min': min(vals),
                'max': max(vals),
                'values': sorted(set(vals)),
                'range_str': get_range(vals)
            }
    print(f"[DEBUG] Доступные диапазоны для {len(variants)} вариантов:")
    for p, r in ranges.items():
        print(f"  {p}: {r['range_str']}")
    return ranges

# === КОНСОЛЬНЫЙ РЕЖИМ ===
def console_mode():
    current_variants = all_variants.copy()
    targets = {}
    remaining_params = PARAMS.copy()

    print("СИСТЕМА УМНОЙ РЕКОМЕНДАЦИИ")
    print("=" * 70)

    while remaining_params:
        print(f"\nДОСТУПНЫЕ ПАРАМЕТРЫ ({len(current_variants)} вариантов):")
        print("-" * 50)
        
        ranges = get_param_ranges(current_variants)
        
        for i, p in enumerate(remaining_params, 1):
            print(f"[{i}] {p} → [{ranges[p]['range_str']}]")

        choice = input("\nНомер параметра (или Enter для завершения): ").strip()
        if not choice:
            break

        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(remaining_params)):
                raise ValueError
            param = remaining_params[idx]
        except:
            print("Неверный номер!")
            continue

        target_input = input(f"Цель {param} [{ranges[param]['range_str']}]: ").strip()
        if not target_input:
            remaining_params.pop(idx)
            continue

        try:
            target = float(target_input)
            min_v, max_v = ranges[param]['min'], ranges[param]['max']
            if target < min_v or target > max_v:
                target = max(min_v, min(target, max_v))
                print(f"[INFO] Скорректировано до ближайшего: {target:.2f}")
        except:
            print("Неверное число!")
            continue

        targets[param] = target
        current_variants = filter_variants(current_variants, param, target)
        
        if not current_variants:
            print("[ERROR] НЕВОЗМОЖНО! Нет вариантов с таким значением.")
            break

        remaining_params.pop(idx)

        top3 = get_top3(current_variants, targets)
        print("\nТОП-3 ВАРИАНТА:")
        table = [[f"{pred[p]:.2f}" if p != 'Мин. N' else f"{pred[p]:.1f}" for p in PARAMS] for _, _, pred in top3]
        print(tabulate(table, headers=PARAMS, tablefmt="grid"))
        print()
        for i, (_, combo, pred) in enumerate(top3, 1):
            actions = " | ".join(f"{k}: {'да' if v else 'нет'}" for k, v in zip(action_names, combo))
            print(f"#{i}: {actions}")
        print()

    if current_variants:
        best_combo, best_pred = min(current_variants, key=lambda x: sum(abs(x[1][p] - targets.get(p, x[1][p])) for p in PARAMS))
        print("\nФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ:")
        actions = " | ".join(f"{k}: {'да' if v else 'нет'}" for k, v in zip(action_names, best_combo))
        print(actions)
        print({k: f"{v:.2f}" if k != 'Мин. N' else f"{v:.1f}" for k, v in best_pred.items()})
    else:
        print("Цель недостижима.")

# === ВЕБ-РЕЖИМ (Streamlit) ===
def web_mode():
    import streamlit as st
    import pandas as pd
    
    st.set_page_config(page_title="Почвенный ИИ", layout="centered")
    st.title("🌱 Моделирование свойств почв по реакции на агротехнические и биотехнологические воздействия")

    # Инициализация состояния
    if 'current_variants' not in st.session_state:
        with st.spinner('🔄 Загрузка модели и расчёт начальных вариантов...'):
            st.session_state.current_variants = all_variants.copy()
            st.session_state.targets = {}
            st.session_state.selected_param = None
            st.session_state.step = 0
            print(f"[DEBUG WEB] Инициализация: {len(all_variants)} вариантов загружено")

    current_variants = st.session_state.current_variants
    targets = st.session_state.targets
    remaining_params = [p for p in PARAMS if p not in targets]

    print(f"[DEBUG WEB] Шаг {st.session_state.step}: {len(current_variants)} вариантов, осталось параметров: {remaining_params}")

    # Финальные рекомендации
    if not remaining_params:
        st.success("✅ ГОТОВО! Рекомендации построены")
        top3 = get_top3(current_variants, targets)
        
        print("[DEBUG WEB] Финальный топ-3:")
        for i, (score, combo, pred) in enumerate(top3, 1):
            print(f"  #{i}: score={score:.2f}, pred={pred}")
        
        # Таблица топ-3
        st.markdown("### 🏆 ТОП-3 ВАРИАНТА:")
        table_data = []
        for i, (_, combo, pred) in enumerate(top3, 1):
            table_data.append({
                '#': i,
                'Сорг.%': f"{pred['Сорг.%']:.2f}",
                'Пористость': f"{pred['Пористость']:.2f}",
                'Мин. N': f"{pred['Мин. N']:.1f}"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, hide_index=True, width='stretch')
        
        # Детальное описание каждого варианта
        st.markdown("---")
        for i, (_, combo, pred) in enumerate(top3, 1):
            actions = " | ".join(f"**{k}**: {'да' if v else 'нет'}" for k, v in zip(action_names, combo))
            st.markdown(f"**#{i}:** {actions}")
        
        st.markdown("---")
        
        # Развернутые карточки
        for i, (_, combo, pred) in enumerate(top3, 1):
            with st.expander(f"🔬 Детали варианта {i}"):
                cols = st.columns(5)
                for j, (name, val) in enumerate(zip(action_names, combo)):
                    emoji = "✅" if val else "❌"
                    cols[j].markdown(f"**{name}**<br>{emoji}", unsafe_allow_html=True)
                
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Сорг.%", f"{pred['Сорг.%']:.2f}")
                col2.metric("Пористость", f"{pred['Пористость']:.2f}")
                col3.metric("Мин. N", f"{pred['Мин. N']:.1f}")
        
        if st.button("🔄 Начать заново", type="primary"):
            st.session_state.clear()
            st.rerun()
        return

    # ШАГ 1: Выбор параметра
    if st.session_state.selected_param is None:
        st.markdown(f"### Шаг {st.session_state.step + 1} из {len(PARAMS)}: Выбор параметра")
        st.info(f"📊 Доступно вариантов: **{len(current_variants)}**")
        
        # Показываем информацию о пересчёте
        with st.spinner('🔄 Расчёт доступных диапазонов...'):
            ranges = get_param_ranges(current_variants)
        
        # Показываем кнопки для каждого параметра
        cols = st.columns(len(remaining_params))
        for i, param in enumerate(remaining_params):
            with cols[i]:
                st.markdown(f"**{param}**")
                st.caption(f"Диапазон: {ranges[param]['range_str']}")
                if st.button(f"Выбрать", key=f"btn_{param}", width='stretch'):
                    st.session_state.selected_param = param
                    print(f"[DEBUG WEB] Выбран параметр: {param}")
                    st.rerun()
        
        st.progress(st.session_state.step / len(PARAMS))
        return

    # ШАГ 2: Ввод значения выбранного параметра
    param = st.session_state.selected_param
    ranges = get_param_ranges(current_variants)
    param_range = ranges[param]
    
    st.markdown(f"### Шаг {st.session_state.step + 1} из {len(PARAMS)}: {param}")
    st.info(f"📊 **Допустимый диапазон:** `{param_range['range_str']}`")
    
    min_v = param_range['min']
    max_v = param_range['max']
    
    # Защита от min = max
    if abs(max_v - min_v) < 0.01:
        st.warning(f"⚠️ Доступно только одно значение: **{min_v:.2f}**")
        
        col1, col2 = st.columns(2)
        
        if col1.button("✅ Применить это значение", type="primary", width='stretch'):
            print(f"[DEBUG WEB] Применяем единственное значение {param}={min_v}")
            
            with st.spinner(f'🔄 Пересчёт вариантов для {param} = {min_v}...'):
                targets[param] = min_v
                new_variants = filter_variants(current_variants, param, min_v)
            
            st.session_state.current_variants = new_variants
            st.session_state.targets = targets
            st.session_state.selected_param = None
            st.session_state.step += 1
            st.success(f"✅ Осталось вариантов: {len(new_variants)}")
            st.rerun()
        
        if col2.button("⏭️ Пропустить параметр", width='stretch'):
            st.session_state.selected_param = None
            st.session_state.step += 1
            st.rerun()
        return
    
    step = 0.1 if param != 'Мин. N' else 0.5
    target = st.slider(
        "Выберите целевое значение",
        min_value=float(min_v),
        max_value=float(max_v),
        value=float((min_v + max_v) / 2),
        step=float(step),
        key=f"slider_{param}_{st.session_state.step}"
    )

    col1, col2, col3 = st.columns(3)
    
    if col1.button("✅ Применить", type="primary", width='stretch'):
        print(f"[DEBUG WEB] Применяем {param}={target}")
        
        # Показываем окно обработки
        with st.spinner(f'🔄 Пересчёт вариантов для {param} = {target}...'):
            targets[param] = target
            new_variants = filter_variants(current_variants, param, target)
        
        if not new_variants:
            st.error("⚠️ С таким значением нет вариантов! Попробуйте другое.")
        else:
            st.session_state.current_variants = new_variants
            st.session_state.targets = targets
            st.session_state.selected_param = None
            st.session_state.step += 1
            st.success(f"✅ Осталось вариантов: {len(new_variants)}")
            st.rerun()

    if col2.button("🔙 Другой параметр", width='stretch'):
        st.session_state.selected_param = None
        st.rerun()
    
    if col3.button("⏭️ Пропустить", width='stretch'):
        st.session_state.selected_param = None
        st.session_state.step += 1
        st.rerun()

    # Прогресс
    st.progress(st.session_state.step / len(PARAMS))

# === ЗАПУСК ===
if __name__ == '__main__':
    if is_streamlit():
        web_mode()
    else:
        console_mode()
