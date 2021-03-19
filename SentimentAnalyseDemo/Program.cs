using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SentimentAnalyseDemo
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
    class Program
    {
        private static readonly string DataPath = @"C:/temp/yelp_labelled.txt";
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);
            var sentimentModel = BuildSentimentModel(mlContext);

            var opinions = new List<SentimentData>
            {
                new SentimentData {SentimentText = "This is an awful!"},
                new SentimentData {SentimentText = "This is excellent!"},
                new SentimentData {SentimentText = "Service was very prompt."},
                new SentimentData {SentimentText = "This was like the final blow!"},
            };

            PredictSentiment(mlContext, sentimentModel, opinions);

            Console.ReadLine();
        }

        private static ITransformer BuildSentimentModel(MLContext mlContext)
        {
            var data = mlContext.Data.LoadFromTextFile<SentimentData>(path: DataPath, hasHeader: true, separatorChar: '\t');

            var dataProcessPipeLine = mlContext.Transforms.Text
                .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText));

            var trainingPipeLine = dataProcessPipeLine
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            var cvResults = mlContext.BinaryClassification
               .CrossValidate(data, estimator: trainingPipeLine);

            var accs = cvResults.Select(r => r.Metrics.Accuracy);
            var auc = cvResults.Select(r => r.Metrics.AreaUnderRocCurve);
            var f1s = cvResults.Select(r => r.Metrics.F1Score);
            var model = trainingPipeLine.Fit(data);

            Console.WriteLine("Bewertung der Qualitätsmetriken des Modells:");
            Console.WriteLine($"Accuracy: {accs.Average():P2}");
            Console.WriteLine($"Auc: {auc.Average():P2}");
            Console.WriteLine($"F1Score: {f1s.Average():P2}");
            Console.WriteLine("=============================================");

            return model;
        }

        private static void PredictSentiment(MLContext mlContext, ITransformer sentimentModel, List<SentimentData> opinions)
        {
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(sentimentModel);

            Console.WriteLine("\nText     | Prognose | Wahrscheinlichkeit positiv");
            foreach (var item in opinions)
            {
                var resultprediction = predEngine.Predict(item);
                var predSentiment = Convert
                    .ToBoolean(resultprediction.Prediction)

                                      ? "positiv" : "negativ";

                Console.WriteLine("{0} | {1} | {2}",
                    item.SentimentText, predSentiment, resultprediction.Probability);
            }
        }
    }    
}
