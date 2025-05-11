Natural disasters, particularly floods, are becoming increasingly frequent and severe, posing
significant risks to infrastructure, communities, and economic systems. Flooding not only
disrupts daily life but also impacts local real estate markets, as property values often decrease in
affected areas. Understanding and quantifying the economic consequences of floods is critical for
stakeholders such as governments, property investors, and homeowners, who need to assess risk
and plan for resilience. However, reliable forecasting of such events and understanding of their
long-term effects remains a challenging problem due to the complexity of environmental data and
difficulty predicting future anomalous weather events such as major floods. Our project explores
whether ML models can help quantify and forecast the economic impact of environmental risks,
focusing on flood events.

This project addresses two interconnected challenges:


1. Predicting heavy rainfall events as a proxy for flood risk: We investigated whether ML
models can predict heavy rainfall events, which serve as a proxy for flood risk. Using decision
tree-based models such as XGBoost and Random Forest, we classify rainfall events and predict
their intensity.

2. Quantifying the impact of flood events on regional housing markets: We examine how flood
events, in terms of their severity and displacement effects, correlate with changes in Housing
Price Indices (HPI) at the ZIP code level. We apply a variety of models, including Random
Forest, CatBoost, and TabPFN, to predict the impact of floods on housing prices.

Additionally, due to the limited availability of data, we use synthetic data generation tech-
niques such as Gaussian Mixture Model (GMM) to improve the models’ performance and
generalization.
