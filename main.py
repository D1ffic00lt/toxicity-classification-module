import warnings

from toxicityclassifier import ToxicityClassificatorV1

warnings.filterwarnings("ignore")

data = input("Enter text (enter empty string to exit): ")
classificator = ToxicityClassificatorV1()
while data != "":
    predictions = classificator.predict(data)
    print("Status: {0} probability: {1}".format(predictions[0], predictions[1]))
    data = input("Enter text (enter empty string to exit): ")
