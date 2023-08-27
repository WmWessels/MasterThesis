# MasterThesis

SBPort is a Python framework for warm-starting AutoML systems. Additionally SBPort, can be used for detached portfolio evaluation for a given dataset. 

Inference includes the following steps:
- Calculate meta-features
- Assignment to a partition
- Return the portfolio attached to the partition
- Either warm-start an AutoML system or directly evaluate the portfolio

# Example Detached Evaluation

```from SBPort.inference import SBPort, PortfolioTransformer
from SBPort.config import inference_kwargs

if __name__=="__main__":
    sbport = SBPort()
    pipeline_results = sbport.run(
        dataset_id = 3,
        inference_pipeline_path="SBPort/optimal_configurations/bin/optimal_bin_max_8_psize_16",
        **inference_kwargs["bin_kwargs"],
    )
    print(pipeline_results)
```
# Example GAMA warm-start

```
from SBPort.inference import SBPort, PortfolioTransformer
from SBPort.meta_features import ClassificationMetaFeatures
from SBPort.config import inference_kwargs
from SBPort.utils import prepare_openml_for_inf, sklearn_to_gama_str
from gama import GamaClassifier

if __name__=="__main__":
    dataset_id = 3
    sbport = SBPort()
    inference_kwargs = inference_kwargs["bin_kwargs"]
    X, y, categorical_indicator = prepare_openml_for_inf(dataset_id)
    metafeatures = sbport.calculate_metafeatures(
        X, 
        y, 
        extractor = ClassificationMetaFeatures,
        numerical_features_with_outliers = inference_kwargs.get("numerical_features_with_outliers"),
        is_clf = inference_kwargs.get("is_clf"),
        is_binary = inference_kwargs.get("is_binary"),
        categorical_indicator=categorical_indicator,
        )
    inference_pipeline = sbport.load_inference_pipeline("SBPort/optimal_configurations/bin/optimal_bin_heuristic_5_psize_16")
    warm_start_sklearn_pipelines = sbport._transform(inference_pipeline, metafeatures)
    gama_formatted_pipelines = [sklearn_to_gama_str(pipeline) for pipeline in warm_start_sklearn_pipelines]

    clf = GamaClassifier(
        max_total_time = 3600, 
        max_eval_time = 360,
        scoring = inference_kwargs.get("scoring")
    )

    clf.fit(X, y, warm_start = gama_formatted_pipelines)

```

