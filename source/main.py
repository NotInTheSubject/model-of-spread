import torch.nn as nn
import torch
from os import sep as os_sep
from collections import namedtuple

# Описываем пути до файлов с информацией

def get_full_path(relative_path):
    import os
    return os.path.dirname(os.path.realpath(__file__)) + os.sep + relative_path


day_as_seconds = 86400

assert os_sep == '/', "there are relative paths using '/' as a path separator, but the current OS use another one"

covid_stat_json_path = get_full_path("../external/mediazona-data/data.json")

wether_csv_paths = {
    "Санкт-Петербург": get_full_path("../external/wheather_set/Piter.02.03.2020.21.12.2020.1.0.0.ru.utf8.00000000.csv"),
    "Москва": get_full_path("../external/wheather_set/Moscow.03.02.2020.19.12.2020.1.0.0.ru.utf8.00000000.csv")}




def foldl(func, acc, iterable):
    for val in iterable:
        acc = func(acc, val)
    return acc


def drop(iterable, n):
    generator = (x for x in iterable)
    junk = [x for x in zip(range(n), generator)]
    return generator


def unzip(zipped):
    zipped_list = list(zipped)
    return tuple(zip(*zipped_list))


def date_as_int(val):
    import datetime
    from dateutil.parser import parse as parse_date
    res = 0

    if isinstance(val, str):
        val = parse_date(val)
    if isinstance(val, datetime.datetime):
        res = int(val.timestamp() // day_as_seconds)
    elif isinstance(val, int):
        res = int(val)
    else:
        raise Exception(
            f"Expect datetime format. input type is {str(type(val))}")
    return res


def int_as_date(val):
    from datetime import datetime

    if not isinstance(val, int):
        raise Exception("Expect int val")
    return datetime.fromtimestamp(val * day_as_seconds)


class Values:
    def __init__(self, start_date):
        if isinstance(start_date, int):
            start_date = int_as_date(start_date)
        if date_as_int(start_date) == None:
            raise Exception(
                f"Cannot use start_date - {str(type(start_date))}." +
                " Expect not datatime type or str which can be converted to datetime")
        self.data = []
        self.start_date = start_date

    def __getitem__(self, arg_day):
        day = date_as_int(arg_day)
        start_date = date_as_int(self.start_date)

        if day < start_date or day > (start_date + len(self.data) - 1):
            raise Exception(f"No value at specified day {str(arg_day)}")
        return self.data[day - start_date]

    def append(self, value):
        return self.data.append(value)

    def items(self):
        return zip(generate_dates_from(self.start_date), self.data)

    def datetimes(self):
        return unzip(self.items())[0]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        from pprint import pformat
        return pformat(self.__dict__)


def to_gamma(t, delta):
    start_date = date_as_int(t.start_date)
    gamma = Values(start_date + 1)

    for day in range(start_date + 1, start_date + len(t) - delta):
        numerator = t[day + delta] - t[day + delta - 1]
        denominator = t[day + delta - 1] - t[day - 1]
        res = numerator / denominator if denominator != 0 \
            else numerator * float('inf')  # note: 0 * inf = nan
        gamma.append(res * delta)
    return gamma


def load_covid_stat(file):
    import json
    from dateutil.parser import parse

    def get_t(history, confirmed):
        prev_val = history[-1] if len(history) != 0 else 0
        new_val = prev_val + confirmed
        return history + [new_val]

    obj = json.load(file)

    start_date = date_as_int(obj['startDate'])
    from datetime import date

    t = {}
    for location in obj['data']:
        v = Values(start_date)
        v.data = foldl(get_t, [], location['confirmed'])
        t[location['name']] = v
    return t


def generate_dates_from(start_date):
    from datetime import datetime as datetime_type
    from datetime import timedelta
    from dateutil.parser import parse as parse_date
    from itertools import count as generator

    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if not isinstance(start_date, datetime_type):
        raise Exception("Expect datetime input")
    return (start_date + timedelta(days=i) for i in generator())


class DataKeeper:
    _instance = namedtuple('DataKeeper_content', 'covid_stat, wheather')

    @staticmethod
    def _init_if_it_is_not():
        with open(covid_stat_json_path, 'r') as file:
            DataKeeper._instance.covid_stat = load_covid_stat(file)

        wheather = {}
        for location, filepath in wether_csv_paths.items():
            with open(filepath, 'r') as file:
                wheather[location] = Wheather.get_values(file)
        DataKeeper._instance.wheather = wheather

    @staticmethod
    def get_whether(location):
        import copy
        DataKeeper._init_if_it_is_not()
        return copy.deepcopy(DataKeeper._instance.wheather[location])

    @staticmethod
    def get_covid_stat(location):
        import copy
        DataKeeper._init_if_it_is_not()
        return copy.deepcopy(DataKeeper._instance.covid_stat[location])


class DayContext:
    def __init__(self):
        self.t


is_pleasant_wheather_map = {
    '': 1,
    'Буря': 0,
    'Гроза (грозы) с осадками или без них.': 0,
    'Дождь со снегом или другими видами твердых осадков': 0,
    'Дождь.': 0,
    'Ливень (ливни).': 0,
    'Метель': 0,
    'Морось.': 0,
    'Облака покрывали более половины неба в течение всего соответствующего периода.': 0,
    'Облака покрывали более половины неба в течение одной части соответствующего периода и половину или менее в течение другой части периода.': 1,
    'Облака покрывали половину неба или менее в течение всего соответствующего периода.': 1,
    'Снег и/или другие виды твердых осадков': 0,
    'Туман или ледяной туман или сильная мгла.': 0}


class Wheather:

    def __init__(self, row):
        Wheather.assert_data_is_expected(row)

        self._row = row
        self._timefield_name = None

        for key, value in row.items():
            if key is not None and key.startswith('Местное время'):
                self._timefield_name = key
        if self._timefield_name is None:
            raise Exception('Cannot find a time keyword')

    @staticmethod
    def get_values(file):
        import csv
        from itertools import groupby

        def round_day(date):
            from datetime import timedelta
            return date - timedelta(hours=date.hour, minutes=date.minute, seconds=date.second)

        rows = csv.DictReader(file, delimiter=';')

        sorted_wheather_data = sorted(
            map(Wheather, rows),
            key=lambda x: x.datetime)

        data = [list(daydata) for _, daydata in groupby(
            sorted_wheather_data,
            lambda x: round_day(x.datetime))]

        assert len(data) != 0, "Input data must contain at list one row"

        assert (data[-1][0].datetime - data[0][0].datetime).days == (len(data) - 1), \
            Exception(f"""
            Unexpected wheather data gaps
            datedelta={(data[-1][0].datetime - data[0][0].datetime).days}
            len(data)={len(data)}""")

        result = Values(start_date=data[0][0].datetime)
        result.data = data
        return result

    @staticmethod
    def assert_data_is_expected(row):
        assert row['W1'] in is_pleasant_wheather_map, f"{row['W1']} is an unexpected value"

    @property
    def is_pleasant(self):
        return is_pleasant_wheather_map[self._row['W1']]

    @property
    def datetime(self):
        from dateutil.parser import parse
        return parse(self._row[self._timefield_name], dayfirst=True)

    @property
    def temperature(self):
        return float(self._row["T"])

    def __str__(self):
        from pprint import pformat
        return pformat(self._row)


def to_wheather_features(wheather_data):
    from datetime import timedelta

    result = Values(start_date=wheather_data.start_date + timedelta(days=1))

    def get_features(date):
        target_date = date - timedelta(days=1)
        daydata = wheather_data[target_date]

        is_pleasant_day = foldl(
            lambda is_pleasant_before, curr: is_pleasant_before and curr.is_pleasant, True, daydata)
        mean_day_temperature = foldl(
            lambda acc_temperature, curr: acc_temperature + curr.temperature, 0., daydata) / len(daydata)
        square_mean_day_temperature = mean_day_temperature ** 2
        return is_pleasant_day, mean_day_temperature, square_mean_day_temperature

    result.data = list(map(get_features, map(
        lambda date: date + timedelta(days=1), wheather_data.datetimes())))
    return result


class Net(nn.Module):
    def __init__(self, features_len):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(
            in_features=features_len, out_features=2, bias=True)
        self.linear2 = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, x):

        y = self.linear1(x)
        y = nn.ELU(0.5)(y)
        y = self.linear2(y)
        return y

 ## Запускаем обучение с преоставленными параметрами
def train(train_features, train_answers, criterion, model=None, **kwargs):

    device = torch.device("cpu")
    if model is None:
        model = Net(features_len=len(train_features[0]))
        model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=kwargs.get("lr", 1e5),
        weight_decay=kwargs.get("weight_decay", 0.01))

    dataset = list(zip(train_features, train_answers))
    batch_size = 30
    batch_count = len(dataset) // batch_size
    train_data = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True)

    for epoch in range(kwargs.get("epoch_count", 15)):
        running_loss = 0
        for batch_index, data in enumerate(train_data):
            inputs, answer = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, answer)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %
              (epoch + 1, running_loss/len(train_data)))

    print('Done Training')
    return model

## Проверяем можель на предоставленных данных
def estimate_model(model, criterion, features, answers):
    loss = 0.
    with torch.no_grad():
        for x, answer in zip(features, answers):
            y = model(x)
            loss += criterion(y.view(-1), answer.view(-1))
    return loss / len(features)

# Получаем, перемешиваем и разделяем (на тестовую и обуч. выборку) датасет
def get_train_test_data(location, train_start_date):
    stat = DataKeeper.get_covid_stat(location)
    wheather_data = DataKeeper.get_whether(location)

    features = to_wheather_features(wheather_data)
    gamma = to_gamma(stat, delta)

    assert features.start_date <= train_start_date
    assert gamma.start_date <= train_start_date

    train_features = Values(start_date=train_start_date)
    train_gamma = Values(start_date=train_start_date)

    train_features.data = list(map(torch.tensor, drop(
        features.data, (train_start_date - features.start_date).days)))
    train_gamma.data = list(map(torch.tensor, drop(
        gamma.data, (train_start_date - gamma.start_date).days)))

    data = list(zip(train_features.data, train_gamma.data))
    random.shuffle(data)
    train_data, test_data = data[:int(len(data)*0.9)], data[int(len(data)*0.9):]
    return train_data, test_data

if __name__ == "__main__":
    from dateutil.parser import parse as parse_date
    import itertools
    import random

    delta = 10

    result_model, result_rating = None, float('inf')
    # location = 'Москва'
    location = "Санкт-Петербург"
    train_start_date = parse_date("01-05-2020", dayfirst=True)

    train_data, test_data = get_train_test_data(location, train_start_date)

    variants = {
        "lr": [2 * x * 1e-4 for x in range(1, 10)],
        "weight_decay": [x * 1e-3 for x in range(10)],
        "epoch_count": [20]
    }
    lr_res, weight_decay_res, epoch_count_res = None, None, None
    for lr, weight_decay, epoch_count in \
            itertools.product(variants["lr"],
                              variants["weight_decay"],
                              variants["epoch_count"]):
        print("Start training")
        print("lr =", lr, "weight_decay =", weight_decay, "epoch_count =", epoch_count)
        criterion = nn.SmoothL1Loss(reduction="mean")
        model = train(
            *unzip(train_data),
            criterion=criterion,
            lr=lr,
            weight_decay=weight_decay,
            epoch_count=epoch_count)
        rating = estimate_model(model, criterion, *unzip(test_data))
        print("rating =", rating)

        if rating < result_rating:
            result_model, result_rating = model, rating
            lr_res, weight_decay_res, epoch_count_res = lr, weight_decay, epoch_count

    print("result_rating =", result_rating)
    print("lr_res =", lr_res)
    print("weight_decay_res =", weight_decay_res)
    print("epoch_count_res =", epoch_count_res)

    torch.save(result_model.state_dict(), \
        f"maybe_good/result_model_{result_rating}_{lr_res}_{weight_decay_res}_{epoch_count_res}.pt")
