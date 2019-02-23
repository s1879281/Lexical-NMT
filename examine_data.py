from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats.stats import pearsonr

class Raw_data():
    def __init__(self, file):
        self.type_dict = defaultdict(lambda: 0)
        self.mod_type_dict = defaultdict(lambda: 0)
        self.sentence_length = []
        self.token_count = 0
        self.token_replace_count = 0
        self._load_data(file)
        self._unk_replace()

    def _load_data(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                token_list = line.strip().split()
                self.sentence_length.append(len(token_list))
                for token in token_list:
                    self.type_dict[token] += 1
                    self.token_count += 1

    def _unk_replace(self):
        for key, value in self.type_dict.items():
            if value != 1:
                self.mod_type_dict[key] = value
            else:
                self.mod_type_dict['unk'] += 1
                self.token_replace_count += 1

def draw(en_length, jp_length):
    fig, axScatter = plt.subplots(figsize=(5, 5))

    # the scatter plot:
    axScatter.scatter(en_length, jp_length, s=0.2)
    plt.xlabel('English sentence length')
    plt.ylabel('Japanese sentence length')

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)

    axHistx.hist(en_length, bins=100)
    axHisty.hist(jp_length, bins=100, orientation='horizontal')

    plt.draw()
    plt.savefig('Q2.pdf')
    plt.show()

if __name__ == '__main__':
    raw_data_en = Raw_data('raw_data/train.en')
    raw_data_jp = Raw_data('raw_data/train.jp')
    print('The token number in English data is: ', raw_data_en.token_count)
    print('The token number in Japanese data is: ', raw_data_jp.token_count)
    print('The type number in English data is: ', len(raw_data_en.type_dict.keys()))
    print('The type number in Japanese data is: ', len(raw_data_jp.type_dict.keys()))
    print('The number of tokens that will be replaced in English data is ', raw_data_en.mod_type_dict['unk'])
    print('The number of tokens that will be replaced in Japanese data is ', raw_data_jp.mod_type_dict['unk'])
    print('Pearson correlation coefficient: ', pearsonr(raw_data_en.sentence_length, raw_data_jp.sentence_length)[0])
    draw(raw_data_en.sentence_length, raw_data_jp.sentence_length)
