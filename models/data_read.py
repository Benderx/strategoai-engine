import pandas
import numpy as np


CSV_LOCATION = 'games'

def import_data():
    return pandas.read_pickle(CSV_LOCATION)


def shuffle(df):     
    print("Shuffling data")
    df = df.sample(frac=1).reset_index(drop=True)
    print("Done shuffling")
    return df


def print_board(board_to_print):
    print()
    for y in range(6):
        arr_temp = []
        for x in range(6):
            val = board_to_print[x + y*6]
            if val == 12:
                name = 'F'
            elif val == 10:
                name = 'B'
            elif val == 11:
                name = 'S'
            elif val == 13:
                name = 'L'
            else:
                name = str(val)
            arr_temp.append(name)
        print(' '.join(map(str, arr_temp)))
    print()


def analyze_chunk(arr, n_percent):
    num_moves = int(n_percent*len(arr))
    print('Analyzing top', n_percent, '% of moves.')
    print(num_moves, '/', len(game_data))
    upper_n = arr[-int(n_percent*len(arr)):]
    print(len(upper_n))

    tot = 0
    for x in upper_n:
        tot += x[1]

    print('Avg samples for top', n_percent, '% of moves')


def main():
    # np.random.seed(seed=4567)

    data = import_data()


    sdata = shuffle(data)

    counter = 0
    game_data = []
    for index, row in sdata.iterrows():
        if counter >= 10000:
            break
        # print('row num', index)
        # print_board(row['board'])
        # # print(row['move_data'])
        # print(row['move_from'] % 6, int(row['move_from']/6))
        # print(row['move_to'] % 6, int(row['move_to']/6))
        # print('rating', row['move_rating'])
        # print('samples', row['move_samples'])
        game_data.append((row['move_rating'], row['samples'], row['board'], row['move_from'], row['move_to']))
        counter += 1

    game_data = sorted(game_data, key=lambda x: x[0])
    
    percent_iter = lambda x: (x/1.0)
    for i in percent_iter:
        analyze_chunk(game_data, i)


main()