# Algo-Trading
"An advanced market sentiment analyzer that uses real-time PCR, Open Interest, and India VIX to interpret market direction. It processes live option chain data, detects traps, volatility shifts, buildups, and trend changes using rule-based logic to generate actionable sentiment insights for traders."


#Description

The Advanced Market Sentiment Analyzer is a data-driven system designed to decode real-time market sentiment using three major derivatives indicators—Put-Call Ratio (PCR), Open Interest (OI), and India VIX (Volatility Index).

Most retail traders depend on a single PCR number, which is often delayed, misinterpreted, or misleading in volatile market conditions. This project aims to solve that limitation by building a holistic sentiment engine that intelligently combines PCR trends, OI structures, and volatility behavior to produce actionable insights.

Using live 5-minute Option Chain data, the system fetches real-time OI, IV, Volume, strike-level data, index prices, and VIX. These inputs are processed through a rule-based logic engine, capable of identifying market scenarios such as:

1.Long/Short build-ups
2.Short covering & long unwinding
3.Bullish or bearish traps (Smart Money detection)
4.False PCR signals
5.Breakout/Breakdown possibilities
6.Overbought/Oversold reversal zones
7.Volatile market "danger" periods
