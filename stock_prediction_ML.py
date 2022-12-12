import comp140_module3 as stocks
import random
import pathlib
import urllib.request
import simpleplot
from collections import defaultdict

class stocks:
    def __init__(self):
        pass

    def get_supported_symbols(self):
        return SYMBOLS

    def get_historical_prices(self, ticker):
        if ticker not in SYMBOLS:
            msg = "No data is available for "
            msg += str(ticker) + " try one of " + str(SYMBOLS)
            raise ValueError(msg)
        name = "stock_prediction_bot" + ticker + ".txt"
        url = pathlib.Path(name).as_uri()
        stockf = urllib.request.urlopen(url)
        stockd = stockf.read().decode('ascii')
        return [float(item) for item in stockd.split()]

    def get_test_prices(self, ticker):
        if ticker not in SYMBOLS:
            msg = "No data is available for "
            msg += str(symbol) + " try one of " + str(SYMBOLS)
            raise ValueError(msg)
        if ticker == "DJIA":
            return DJIA_JAN2013
        elif ticker == "GOOG":
            return GOOG_JAN2013
        elif ticker == "FSLR":
            return FSLR_JAN2013
        # Error.  Shouldn't reach here.
        return []

    def compute_daily_change(self, price_data):
        return [(price_data[i] - price_data[i-1]) / float(price_data[i-1])
            for i in range(1, len(price_data))]

    def bin_daily_changes(self, deltas):
        binned = []
        for val in deltas:
            newbin = -1
            for idx, binval in enumerate(BINS):
                if val < binval:
                    newbin = idx
                    break
            if newbin == -1:
                newbin = len(BINS)
            binned.append(newbin)
        return binned

    def create_histogram(data):
        hist = defaultdict(int)
        for item in data:
            hist[item] += 1
        return hist

    def plot_daily_change(self, changes):
        symbols = list(changes.keys())
        changedata = [list(enumerate(changes[symbol])) for symbol in symbols]
        simpleplot.plot_lines("Daily Change", 800, 400, "Day", "Change", changedata, False, symbols)

    def plot_bin_histogram(self, bins):
        hist = []
        symbols = []
        for symbol, data in bins.items():
            symbols.append(symbol)
            hist.append(create_histogram(data))
        simpleplot.plot_bars("Bin Histogram", 800, 400, "Bin", "Number of Days", hist, symbols)

    

# CREATING A HELPER FUNCTION
# CREATES A ZERO'TH ORDER MARKOV CHAIN
def zero_order_markov(provided_list):
    """
    Create a 0th Order Markov chain with the given list 

    inptus: 
    - a list containing all the possible choices
    
    returns: 
    - A dicitonary containing the possible choice as a 
    """
    dictionary = {}
    for data in provided_list:
        if data in dictionary:
            dictionary[data] = dictionary[data] + 1
        else:
            dictionary[data] = 1
    for element in dictionary:
        dictionary[element] = dictionary[element]/len(provided_list)
    return dictionary

#CREATING THE MODEL
def markov_chain(data, order):
    """
    Create a Markov chain with the given order from the given data.

    inputs:
        - data: a list of ints or floats representing previously collected data
        - order: an integer repesenting the desired order of the markov chain

    returns: a dictionary that represents the Markov chain
    """
    markov = {}
    for element in range(0,len(data)):
        value = []
        counter = order
        while counter > 0:
            if (element + 1 - order) > 0:
                value.append(data[element-counter])
            counter -= 1
        if (value != []) and (tuple(value) in markov) == False:
            markov[tuple(value)] = []
        if value != []:
            markov[tuple(value)].append(data[element])
    for value in markov:
        markov[value] = zero_order_markov(markov[value])
    return markov


# CREATING A FUCNTION TO PREDICT FUTURE PRICES

def predict(model, last, num):
    """
    Predict the next num values given the model and the last values.

    inputs:
        - model: a dictionary representing a Markov chain
        - last: a list (with length of the order of the Markov chain)
                representing the previous states
        - num: an integer representing the number of desired future states

    returns: a list of integers that are the next num states
    """
    final_list = []
    for _ in range(num):
        gen_list = tuple(last)
        nxt = 0
        if gen_list in model:
            dictionary = model[gen_list]
            newdict = {}
            prev = 0
            for gen_list, count in dictionary.items():
                newdict[gen_list] = (prev, prev + count)
                prev += count
            random_number = random.random()
            for gen_list, count in newdict.items():
                if count[0] <= random_number < count[1]:
                    nxt = gen_list
        else:
            nxt = random.choice([0,1,2,3])
        final_list.append(nxt)
        last = last[1:] + [nxt]
    return final_list


### Error

def mse(result, expected):
    """
    Calculate the mean squared error between two data sets.

    The length of the inputs, result and expected, must be the same.

    inputs:
        - result: a list of integers or floats representing the actual output
        - expected: a list of integers or floats representing the predicted output

    returns: a float that is the mean squared error between the two data sets
    """
    num = 0
    for element in range(len(result)):
        num = num + (result[element] - expected[element])**2
    return num/len(result)


### EXPERIMENT

def run_experiment(train, order, test, future, actual, trials):
    """
    Run an experiment to predict the future of the test
    data given the training data.

    inputs:
        - train: a list of integers representing past stock price data
        - order: an integer representing the order of the markov chain
                 that will be used
        - test: a list of integers of length "order" representing past
                stock price data (different time period than "train")
        - future: an integer representing the number of future days to
                  predict
        - actual: a list representing the actual results for the next
                  "future" days
        - trials: an integer representing the number of trials to run

    returns: a float that is the mean squared error over the number of trials
    """
    dictionary = markov_chain(train, order)
    error = 0
    for _ in range(trials):
        prediction = predict(dictionary, test, future)
        error += mse(prediction, actual)
    return error / trials


### Application

def run():
    """
    Run application.

    You do not need to modify any code in this function.  You should
    feel free to look it over and understand it, though.
    """
    # Get the supported stock symbols
    symbols = stocks.get_supported_symbols()

    # Get stock data and process it

    # Training data
    changes = {}
    bins = {}
    for symbol in symbols:
        prices = stocks.get_historical_prices(symbol)
        changes[symbol] = stocks.compute_daily_change(prices)
        bins[symbol] = stocks.bin_daily_changes(changes[symbol])

    # Test data
    testchanges = {}
    testbins = {}
    for symbol in symbols:
        testprices = stocks.get_test_prices(symbol)
        testchanges[symbol] = stocks.compute_daily_change(testprices)
        testbins[symbol] = stocks.bin_daily_changes(testchanges[symbol])

    # Display data
    #   Comment these 2 lines out if you don't want to see the plots
    stocks.plot_daily_change(changes)
    stocks.plot_bin_histogram(bins)

    # Run experiments
    orders = [1, 3, 5, 7, 9]
    ntrials = 500
    days = 5

    for symbol in symbols:
        print(symbol)
        print("====")
        print("Actual:", testbins[symbol][-days:])
        for order in orders:
            error = run_experiment(bins[symbol], order,
                                   testbins[symbol][-order-days:-days], days,
                                   testbins[symbol][-days:], ntrials)
            print("Order", order, ":", error)
        print()

# You might want to comment out the call to run while you are
# developing your code.  Uncomment it when you are ready to run your
# code on the provided data.

run()
