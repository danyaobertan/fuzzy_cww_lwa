import word
import words_model
import random
import lwa
import numpy as np
import streamlit as st


def main():
    st.title("Інтелектуальний інтерфейс для оцінки співробітника колегами")

    # Розділіть сторінку на дві колонки
    col_soft, col_hard = st.columns(2)

    with col_soft:
        st.markdown("## Софт скіли")

        # Софт скіли
        soft_model = words_model.skills_soft
        grades_soft = []

        # Додаємо інтерфейс для софт скілів
        for word_soft in soft_model["words"].keys():
            st.markdown(f"### {word_soft}")
            counts_soft = int(st.text_input(
                f"Кількість людей, які оцінили софт скіли співробітника як {word_soft} ", random.randint(1, 10)))
            grades_soft += [word_soft] * counts_soft

        # Обробка софт скілів
        W_soft = []
        for item_soft in soft_model["words"]:
            W_soft.append(grades_soft.count(item_soft))

        h_soft = min(item_soft["lmf"][-1]
                     for item_soft in soft_model["words"].values())
        m = 50
        intervals_umf_soft = lwa.alpha_cuts_intervals(m)
        intervals_lmf_soft = lwa.alpha_cuts_intervals(m, h_soft)

        res_lmf_soft = lwa.y_lmf(intervals_lmf_soft, soft_model, W_soft)
        res_umf_soft = lwa.y_umf(intervals_umf_soft, soft_model, W_soft)

        res_soft = lwa.construct_dit2fs(
            np.arange(
                *soft_model["x"]), intervals_lmf_soft, res_lmf_soft, intervals_umf_soft, res_umf_soft
        )

        st.title("Результат для софт скілів")
        res_soft.plot()

        sm_soft = []

        for title_soft, fou_soft in soft_model["words"].items():
            sm_soft.append(
                (
                    title_soft,
                    res_soft.similarity_measure(
                        word.Word(None, soft_model["x"],
                                  fou_soft["lmf"], fou_soft["umf"])
                    ),
                    word.Word(
                        title_soft, soft_model["x"], fou_soft["lmf"], fou_soft["umf"]),
                ),
            )

        res_word_soft = max(sm_soft, key=lambda item: item[1])

        st.title("Найбільша подібність для софт скілів")
        res_word_soft[2].plot()

        st.title("Значення та ймовірність для софт скілів")
        st.markdown(f"### {res_word_soft[2]}: {res_word_soft[1]}")

    with col_hard:
        st.markdown("## Хард скіли")

        # Хард скіли
        hard_model = words_model.skills_hard
        grades_hard = []

        # Додаємо інтерфейс для хард скілів
        for word_hard in hard_model["words"].keys():
            st.markdown(f"### {word_hard}")
            counts_hard = int(st.text_input(
                f"Кількість людей, які оцінили хард скіли співробітника як {word_hard} ", random.randint(1, 10)))
            grades_hard += [word_hard] * counts_hard

        # Обробка хард скілів
        W_hard = []
        for item_hard in hard_model["words"]:
            W_hard.append(grades_hard.count(item_hard))

        h_hard = min(item_hard["lmf"][-1]
                     for item_hard in hard_model["words"].values())

        intervals_umf_hard = lwa.alpha_cuts_intervals(m)
        intervals_lmf_hard = lwa.alpha_cuts_intervals(m, h_hard)

        res_lmf_hard = lwa.y_lmf(intervals_lmf_hard, hard_model, W_hard)
        res_umf_hard = lwa.y_umf(intervals_umf_hard, hard_model, W_hard)

        res_hard = lwa.construct_dit2fs(
            np.arange(
                *hard_model["x"]), intervals_lmf_hard, res_lmf_hard, intervals_umf_hard, res_umf_hard
        )

        st.title("Результат для хард скілів")
        res_hard.plot()

        sm_hard = []

        for title_hard, fou_hard in hard_model["words"].items():
            sm_hard.append(
                (
                    title_hard,
                    res_hard.similarity_measure(
                        word.Word(None, hard_model["x"],
                                  fou_hard["lmf"], fou_hard["umf"])
                    ),
                    word.Word(
                        title_hard, hard_model["x"], fou_hard["lmf"], fou_hard["umf"]),
                ),
            )

        res_word_hard = max(sm_hard, key=lambda item: item[1])

        st.title("Найбільша подібність для хард скілів")
        res_word_hard[2].plot()

        st.title("Значення та ймовірність для хард скілів")
        st.markdown(f"### {res_word_hard[2]}: {res_word_hard[1]}")

    # Визначення загальних скілів на основі софт та хард
    st.title("Загальні скіли")

    # Об'єднати оцінки софт та хард скілів
    all_grades = grades_soft + grades_hard
    print(all_grades)
    if len(all_grades) == 0:
        st.warning("Немає даних для побудови загальних скілів.")
    else:
        # Обробка загальних скілів
        general_model = words_model.skills
        W_general = []
        for item_general in general_model["words"]:
            W_general.append(all_grades.count(item_general))

        h_general = min(item_general["lmf"][-1]
                        for item_general in general_model["words"].values())

        intervals_umf_general = lwa.alpha_cuts_intervals(m)
        intervals_lmf_general = lwa.alpha_cuts_intervals(m, h_general)

        res_lmf_general = lwa.y_lmf(
            intervals_lmf_general, general_model, W_general)
        res_umf_general = lwa.y_umf(
            intervals_umf_general, general_model, W_general)

        res_general = lwa.construct_dit2fs(
            np.arange(
                *general_model["x"]), intervals_lmf_general, res_lmf_general, intervals_umf_general, res_umf_general
        )

        st.title("Результат")
        res_general.plot()

        sm_general = []

        for title_general, fou_general in general_model["words"].items():
            sm_general.append(
                (
                    title_general,
                    res_general.similarity_measure(
                        word.Word(None, general_model["x"],
                                  fou_general["lmf"], fou_general["umf"])
                    ),
                    word.Word(
                        title_general, general_model["x"], fou_general["lmf"], fou_general["umf"]),
                ),
            )

        res_word_general = max(sm_general, key=lambda item: item[1])

        st.title("Найбільша подібність для хард скілів")
        res_word_general[2].plot()

        st.title("Значення та ймовірність для хард скілів")
        st.markdown(f"### {res_word_general[2]}: {res_word_general[1]}")


if __name__ == "__main__":
    main()
