import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Helper function to style the pages
def style_page(title):
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    st.markdown("---")

# Helper function for combinations, used in Binomial distribution
def combinations(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

# Introduction to Statistics and Probability
def introduction_page():
    style_page("Introduction to Statistics and Probability")
    st.markdown("""
    Welcome to this interactive guide to Statistics and Probability! This application is designed to provide you with a comprehensive overview of key concepts, from the very basics to more advanced topics.

    ### What is Statistics?
    Statistics is the science of collecting, analyzing, interpreting, presenting, and organizing data. It allows us to make sense of the vast amounts of information we encounter in the world and to draw meaningful conclusions from it.

    ### What is Probability?
    Probability is the measure of the likelihood that an event will occur. It is a fundamental concept in statistics and is used to quantify uncertainty.

    **Use the navigation on the left to explore the different sections of this guide.**
    """)

# Descriptive Statistics
def descriptive_statistics_page():
    style_page("Descriptive Statistics")
    st.header("Measures of Central Tendency")
    
    data = st.text_area("Enter a list of numbers (comma-separated):", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5")
    try:
        num_data = [float(x.strip()) for x in data.split(',')]
        df = pd.DataFrame(num_data, columns=["value"])

        mean = df['value'].mean()
        median = df['value'].median()
        mode = df['value'].mode().to_list()

        st.write(f"**Mean:** {mean:.2f}")
        st.write(f"**Median:** {median:.2f}")
        st.write(f"**Mode:** {', '.join(map(str, mode))}")

        st.header("Measures of Dispersion")
        range_val = df['value'].max() - df['value'].min()
        variance = df['value'].var(ddof=0) # Population variance
        std_dev = df['value'].std(ddof=0)  # Population std dev

        st.write(f"**Range:** {range_val:.2f}")
        st.write(f"**Variance:** {variance:.2f}")
        st.write(f"**Standard Deviation:** {std_dev:.2f}")

        st.header("Data Visualization")
        fig, ax = plt.subplots()
        ax.hist(df['value'], bins='auto', alpha=0.7, rwidth=0.85, density=True)
        ax.set_title("Histogram of Your Data")
        st.pyplot(fig)

    except (ValueError, IndexError):
        st.error("Please enter a valid list of comma-separated numbers.")

# Probability
def probability_page():
    style_page("Probability")
    st.header("Binomial Distribution")
    
    n = st.slider("Number of trials (n):", 1, 50, 10)
    p = st.slider("Probability of success (p):", 0.0, 1.0, 0.5)

    x = np.arange(0, n + 1)
    # Calculate PMF manually
    pmf = [combinations(n, k) * (p**k) * ((1-p)**(n-k)) for k in x]

    fig, ax = plt.subplots()
    ax.bar(x, pmf)
    ax.set_title("Binomial Distribution PMF")
    ax.set_xlabel("Number of Successes")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    st.header("Normal Distribution")
    mu = st.slider("Mean (μ):", -10.0, 10.0, 0.0, key='mu')
    sigma = st.slider("Standard Deviation (σ):", 0.1, 10.0, 1.0, key='sigma')

    x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    # Calculate PDF manually
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma)**2)

    fig_norm, ax_norm = plt.subplots()
    ax_norm.plot(x_norm, pdf)
    ax_norm.set_title("Normal Distribution PDF")
    ax_norm.set_xlabel("Value")
    ax_norm.set_ylabel("Probability Density")
    st.pyplot(fig_norm)

# Inferential Statistics
def inferential_statistics_page():
    style_page("Inferential Statistics")
    st.header("Confidence Intervals")
    st.markdown("Here we calculate the confidence interval for the mean of a sample. For large samples (n > 30), we can approximate the t-distribution with the standard normal distribution (z-distribution).")
    
    # Generate some data to work with
    data_inf = np.random.normal(loc=20, scale=5, size=100)
    
    z_scores = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    confidence_level = st.selectbox("Confidence Level:", list(z_scores.keys()), index=1)
    
    z_score = z_scores[confidence_level]
    mean_inf = np.mean(data_inf)
    # Standard error of the mean = sample_std / sqrt(n)
    sem = np.std(data_inf, ddof=1) / np.sqrt(len(data_inf))
    interval = z_score * sem

    st.write(f"A **{confidence_level*100:.0f}%** confidence interval for the mean is: **({mean_inf - interval:.2f}, {mean_inf + interval:.2f})**")

    st.header("Hypothesis Testing (t-test)")
    st.write("We will test if the mean of a sample is significantly different from a given value by calculating the t-statistic.")
    
    sample_mean = st.number_input("Sample Mean:", value=5.5)
    sample_std = st.number_input("Sample Standard Deviation:", value=1.2)
    sample_size = st.number_input("Sample Size:", value=30, min_value=2)
    pop_mean = st.number_input("Population Mean (to test against):", value=5.0)

    # Calculate t-statistic manually
    t_stat = (sample_mean - pop_mean) / (sample_std / np.sqrt(sample_size))
    
    st.write(f"**t-statistic:** {t_stat:.2f}")
    st.info("A p-value cannot be calculated without a stats library. However, as a rule of thumb for a sample size of 30, a t-statistic with an absolute value greater than ~2.04 indicates a statistically significant result at the p < 0.05 level.")

# Advanced Topics
def advanced_topics_page():
    style_page("Advanced Topics")
    st.header("Linear Regression")
    st.write("We will create a simple linear regression model from scratch.")

    # Generate some sample data
    X = np.linspace(0, 10, 100)
    y = 2.5 * X + np.random.normal(0, 2, 100)
    
    # Manual calculation of regression coefficients
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean)**2)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Manual calculation of R-squared
    y_pred = slope * X + intercept
    ss_total = np.sum((y - y_mean)**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)

    st.write(f"**Slope:** {slope:.2f}")
    st.write(f"**Intercept:** {intercept:.2f}")
    st.write(f"**R-squared:** {r_squared:.2f}")

    fig_reg, ax_reg = plt.subplots()
    ax_reg.scatter(X, y, label='Original data')
    ax_reg.plot(X, y_pred, 'r', label='Fitted line')
    ax_reg.legend()
    st.pyplot(fig_reg)

    st.header("ANOVA (Analysis of Variance)")
    st.write("ANOVA is used to compare the means of two or more groups by calculating the F-statistic.")
    
    # Sample data for ANOVA
    group1 = np.random.normal(10, 2, 30)
    group2 = np.random.normal(12, 2, 30)
    group3 = np.random.normal(10.5, 2, 30)
    data_anova = [group1, group2, group3]
    
    # Manual calculation of F-statistic
    k = len(data_anova)
    N = sum(len(g) for g in data_anova)
    grand_mean = np.mean(np.concatenate(data_anova))
    
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in data_anova)
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in data_anova)
    
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)
    
    f_stat = ms_between / ms_within
    
    st.write(f"**F-statistic:** {f_stat:.2f}")
    st.info("A p-value is not calculated here. A larger F-statistic suggests that the variation between group means is large relative to the variation within the groups, pointing to a significant difference between the groups.")

# Main app navigation
def main():
    pages = {
        "Introduction": introduction_page,
        "Descriptive Statistics": descriptive_statistics_page,
        "Probability": probability_page,
        "Inferential Statistics": inferential_statistics_page,
        "Advanced Topics": advanced_topics_page,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
