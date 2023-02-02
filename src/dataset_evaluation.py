import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from data_source import *
from pipeline import *
from plot_utils import *
from src.utils import StoreDictKeyPair

IMAGE_TYPE_MAP = {
    (True, False, True): 'Unclassified Data',
    (False, True, True): 'Dry Land',
    (False, False, False): 'Relevant Flood Data',
    (False, False, True): 'No Flood Data',
}
IMAGE_TYPE_COLOR_MAP = {k: cmap[i] for i, k in enumerate(IMAGE_TYPE_MAP.keys())}

IMAGE_TYPE_MAP_ALL = IMAGE_TYPE_MAP.copy()
for perm in itert.combinations([True, False], 3):
    if perm not in IMAGE_TYPE_MAP_ALL:
        IMAGE_TYPE_MAP_ALL[perm] = 'Illegal State'

def percentage_display_total(pct: float):
    return f'{pct:.2f}%'

def plot_flood_no_flood_count(count_axis: plt.Axes, df: pd.DataFrame, add_title: bool):
    # see https://matplotlib.org/stable/gallery/pie_and_polar_charts/nested_pie.html
    # and https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_demo2.html
    count_dict = df.drop(columns=['region']).groupby('data').sum().to_dict()['count']
    no_data_dry_flood_colors = [FLOOD_COLORS[flood_state] for flood_state in count_dict.keys()]
    vals = np.array(list(count_dict.values())).flatten()
    wedges_outer, texts_outer, autotexts_inner = count_axis.pie(vals,
                                                                radius=1, colors=no_data_dry_flood_colors,
                                                                wedgeprops=dict(edgecolor='w'),
                                                                autopct=percentage_display_total,
                                                                labels=list(FLOOD_LABELS.values())[
                                                                       len(FLOOD_LABELS) - vals.shape[0]:])

    plt.setp(texts_outer, visible=False)
    plt.setp(autotexts_inner, size=16, weight='bold', color='w')
    if add_title:
        count_axis.set_title(f'Flood-No-Flood {"" if split is None else split + "-set "}distribution')
    count_axis.legend()

def plot_flood_image_percentage(flood_img_axis: plt.Axes, flood_images: pd.DataFrame, add_title: bool):
    flooded_series: pd.Series = flood_images.drop(columns=['region']).groupby(
        ['is_no_data_only', 'is_dry_only', 'is_no_flood']).size()
    labels = [IMAGE_TYPE_MAP_ALL[label] for label in flooded_series.index.values]
    flood_image_colors = [IMAGE_TYPE_COLOR_MAP[key] for key in flooded_series.index.values]
    flood_no_flood_series = flood_images.drop(columns=['is_no_data_only', 'is_dry_only', 'region']).groupby(
        ['is_no_flood']).size()
    wedges_outer, _, autotexts_inner = flood_img_axis.pie(
        flood_no_flood_series,
        radius=1, colors=cm.get_cmap('tab20c')(np.array([0, 3])),
        autopct=percentage_display_total,
        wedgeprops=dict(edgecolor='w', width=0.3))  # ,
    # labels=list(IMAGE_TYPE_MAP.values())[2:] if len(flood_no_flood_series) > 1 else list(IMAGE_TYPE_MAP.values())[2:3])
    wedges_outer, texts = flood_img_axis.pie(
        flooded_series,
        radius=1 - 0.3, colors=flood_image_colors,
        wedgeprops=dict(edgecolor='w', width=0.45),
        labels=labels)

    plt.setp(autotexts_inner, size=16, weight='bold', color='w')
    plt.setp(texts, size=12, weight='bold', visible=False)
    flood_img_axis.legend()
    if add_title:
        flood_img_axis.set_title(f'Percentage of images with floods{"" if split is None else " in " + split + "-set"}')

def plot_region_distribution(dist_axis: plt.Axes, per_region: pd.DataFrame, region_colors: Dict[str, Any], add_title: bool):
    region_order = per_region['region']
    color_map = cm.get_cmap('Set3')
    for region in sorted(per_region['region']):
        if region not in region_colors:
            region_colors[region] = color_map([len(region_colors)])[0]
    ordered_colors = [region_colors[region] for region in region_order]

    x_positions = np.arange(per_region.shape[0])
    total_sum: float = float(per_region['count'].sum())
    dist_axis.bar(x_positions,
                  height=100.0 * per_region['count'] / total_sum,
                  width=0.4, color=ordered_colors)
    dist_axis.set_xticks(x_positions)
    dist_axis.set_xticklabels(region_order, rotation=45, ha='right')
    dist_axis.set_xlabel('Region')
    dist_axis.set_ylabel('Fraction of Dataset in %' if split is None else 'Fraction of Split in %')
    if add_title:
        dist_axis.set_title(f'Region distribution{"" if split is None else " in " + split + "-set "}')

# fsbr = flood_state_by_region
def plot_fsbr(fsbr_axis: plt.Axes, df: pd.DataFrame, per_region: pd.DataFrame, add_title: bool):
    region_order = per_region['region']
    x_positions = np.arange(per_region.shape[0])
    width = 0.3
    for flood_state in [-1, 0, 1]:
        region_wise_counts = df.loc[df['data'] == flood_state, ['region', 'count']]
        relevant_counts = [float(region_wise_counts.loc[region_wise_counts['region'] == region, ['count']].sum()) /
                           per_region.loc[region_order == region, ['count']].sum()
                           for region in region_order]
        relevant_counts = np.array(relevant_counts).flatten()
        assert relevant_counts.shape[0] == x_positions.shape[0]
        fsbr_axis.bar(x_positions + width * flood_state,
                      height=100.0 * relevant_counts,
                      width=width, color=FLOOD_COLORS[flood_state],
                      label=FLOOD_LABELS[flood_state])
    fsbr_axis.set_xticks(x_positions)
    fsbr_axis.set_xticklabels(region_order, rotation=45, ha='right')
    fsbr_axis.set_xlabel('Region')
    fsbr_axis.set_ylabel('Fraction of Region in %')
    if add_title:
        fsbr_axis.set_title(
            f'Flood state distribution by region distribution{"" if split is None else " in " + split + "-set"}')
    fsbr_axis.legend()

def plot_fibr(fibr_axis: plt.Axes, flood_images: pd.DataFrame, per_region: pd.DataFrame, add_title: bool):
    region_order = per_region['region']
    x_positions = np.arange(per_region.shape[0])
    width = 0.2
    grouped_by_flooded: pd.DataFrame = flood_images.groupby(['is_no_data_only', 'is_dry_only', 'is_no_flood', 'region']) \
        .size().reset_index(name='count')
    num_per_region: pd.DataFrame = flood_images.groupby(['region']).size().reset_index(name='count')
    for i, (descriptor, label) in enumerate(IMAGE_TYPE_MAP.items()):
        if not np.any(grouped_by_flooded[['is_no_data_only', 'is_dry_only', 'is_no_flood']] == descriptor):
            continue
        relevant_counts = [(grouped_by_flooded.loc[
                                np.all(grouped_by_flooded[['is_no_data_only', 'is_dry_only', 'is_no_flood', 'region']]
                                       == (descriptor + (region,)), axis=1), 'count']
                            .sum() / num_per_region.loc[num_per_region['region'] == region, 'count'].sum())
                           for region in region_order]
        relevant_counts = np.array(relevant_counts).flatten()
        assert relevant_counts.shape[0] == x_positions.shape[0]
        fibr_axis.bar(x_positions + i * width - 1.5 * width,
                      height=100.0 * relevant_counts,
                      width=width, color=IMAGE_TYPE_COLOR_MAP[descriptor],
                      label=label)
    fibr_axis.set_xticks(x_positions)
    fibr_axis.set_xticklabels(region_order, rotation=45, ha='right')
    fibr_axis.set_xlabel('Region')
    fibr_axis.set_ylabel('Fraction of Region in %')
    if add_title:
        fibr_axis.set_title(f'Percentage of images with Floods by region{"" if split is None else " in " + split + "-set"}')
    fibr_axis.legend()

def plot_distribution(split: str, df: pd.DataFrame, flood_images: pd.DataFrame, axes: np.ndarray, region_colors: Dict[str, Any],
                      per_region: pd.DataFrame, args: argparse.Namespace) \
        -> Dict[str, Any]:

    count_axis: plt.Axes = axes[0]
    plot_flood_no_flood_count(count_axis, df, args.with_figure_title)
    if args.single_flood_no_flood:
        fig, axs, ld = create_figure_with_subplots('', 1, 1, with_title=args.with_figure_title)
        plot_flood_no_flood_count(axs[0,0], df, args.with_figure_title)
        save_and_show(fig, path.join(args.plot_folder, f'Flood_No_Flood_{split}'), args.show, ld, args.save,
                      with_title=args.with_figure_title)

    flood_img_axis: plt.Axes = axes[1]
    plot_flood_image_percentage(flood_img_axis, flood_images, args.with_figure_title)
    if args.flood_image_percentage:
        fig, axs, ld = create_figure_with_subplots('', 1, 1, with_title=args.with_figure_title)
        plot_flood_image_percentage(axs[0,0], flood_images, args.with_figure_title)
        save_and_show(fig, path.join(args.plot_folder, f'Flood_Image_Percentage_{split}'), args.show, ld, args.save,
                      with_title=args.with_figure_title)

    dist_axis: plt.Axes = axes[2]
    plot_region_distribution(dist_axis, per_region, region_colors, args.with_figure_title)
    if args.single_region_distribution:
        fig, axs, ld = create_figure_with_subplots('', 1, 1, with_title=args.with_figure_title)
        plot_region_distribution(axs[0,0], per_region, region_colors, args.with_figure_title)
        save_and_show(fig, path.join(args.plot_folder, f'Region_Distribution_{split}'), args.show, ld, args.save,
                      with_title=args.with_figure_title)

    fsbr_axis: plt.Axes = axes[3]
    plot_fsbr(fsbr_axis, df, per_region, args.with_figure_title)
    if args.single_fsbr:
        fig, axs, ld = create_figure_with_subplots('', 1, 1, with_title=args.with_figure_title)
        plot_fsbr(axs[0,0],  df, per_region, args.with_figure_title)
        save_and_show(fig, path.join(args.plot_folder, f'FSBR_{split}'), args.show, ld, args.save,
                      with_title=args.with_figure_title)

    fibr_axis: plt.Axes = axes[4]
    plot_fibr(fibr_axis, flood_images, per_region, args.with_figure_title)
    if args.single_fibr:
        fig, axs, ld = create_figure_with_subplots('', 1, 1, with_title=args.with_figure_title)
        plot_fibr(axs[0,0], flood_images, per_region, args.with_figure_title)
        save_and_show(fig, path.join(args.plot_folder, f'FIBR_{split}'), args.show, ld, args.save,
                      with_title=args.with_figure_title)
    return region_colors

def create_per_region_df(df: pd.DataFrame):
    per_region: pd.DataFrame = df.drop(columns=['data']).groupby('region').sum().reset_index('region')
    per_region = per_region.sort_values(by=['count', 'region'], ascending=False)
    return per_region


def plot_set_distribution(split: str, axes: np.ndarray, dataset: data.Dataset, meta: Meta, region_colors: Dict[str, Any]
                          , args: argparse.Namespace) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    regions = [info.region for info in meta.per_item_info]
    unique_regions = list(sorted(set(regions)))
    if isinstance(dataset, ArrayInMemoryDataset):
        # Legacy code based on loading the whole thing into memory...
        data = dataset.data
        cat_type = pd.Categorical([region for region in regions for _ in range(data.shape[-2]) for _ in range(data.shape[-1])],
                                  categories=unique_regions, ordered=True)
        df = pd.DataFrame({'data': data.reshape(-1), 'region': pd.Series(cat_type, name='region')})
        df: pd.DataFrame = df.groupby(['region', 'data']).size().reset_index(name='count')
        flood_images: pd.DataFrame = pd.DataFrame({'is_dry_only': [np.all(ar == 0) for ar in data],
                                                   'is_no_data_only': [np.all(ar == -1) for ar in data],
                                                   'is_no_flood': [np.all(ar != 1) for ar in data],
                                                   'region': pd.Series(pd.Categorical(regions, categories=unique_regions, ordered=True))})
    else:
        region_class_tuples = []
        flood_image_tuples = []
        for img, info in zip(dataset, meta.per_item_info):
            per_item_info: PerItemMeta = info
            is_dry_only, is_no_data_only, is_no_flood = True, True, True
            for clazz in [-1, 0, 1]:
                num_with_clazz = np.count_nonzero(img == clazz)
                if num_with_clazz > 0:
                    region_class_tuples.append((per_item_info.region, clazz, num_with_clazz))
                    is_dry_only = is_dry_only and clazz == 0
                    is_no_data_only = is_no_data_only and clazz == -1
                    is_no_flood = is_no_flood and clazz != 1
            flood_image_tuples.append((is_dry_only, is_no_data_only, is_no_flood, per_item_info.region))
        cat_type = pd.Categorical([region for region, _, _ in region_class_tuples], categories=unique_regions,
                                  ordered=True)
        df: pd.DataFrame = pd.DataFrame({'region': pd.Series(cat_type, name='region'),
                                         'data': [clazz for _, clazz, _ in region_class_tuples],
                                         'count': [count for _, _, count in region_class_tuples]})
        cat_type = pd.Categorical([region for _, _, _, region in flood_image_tuples], categories=unique_regions,
                                  ordered=True)
        flood_images: pd.DataFrame = pd.DataFrame({
            'is_dry_only': [is_dry_only for is_dry_only, _, _, _ in flood_image_tuples],
            'is_no_data_only': [is_no_data_only for _, is_no_data_only, _, _ in flood_image_tuples],
            'is_no_flood': [is_no_flood for _, _, is_no_flood, _ in flood_image_tuples],
            'region': pd.Series(cat_type, name='region')
        })

    region_colors = plot_distribution(split, df, flood_images,  axes, region_colors, create_per_region_df(df), args)
    return region_colors, df, flood_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('plot_folder')
    parser.add_argument('--sci-analysis', dest='include_analysis', action='store_true')
    parser.add_argument('--no-sci-analysis', dest='include_analysis', action='store_false')
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--hide', dest='show', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.add_argument('--rc_params', dest='rc_params', action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL",
                        default={'font.family': 'arial', 'font.size': 11})
    parser.add_argument('--with_figure_title', dest='with_figure_title', action='store_true')
    parser.add_argument('--no-figure_title', dest='with_figure_title', action='store_false')
    parser.add_argument('--single_flood_no_flood', dest='single_flood_no_flood', action='store_true')
    parser.add_argument('--no-single_flood_no_flood', dest='single_flood_no_flood', action='store_false')
    parser.add_argument('--flood_image_percentage', dest='flood_image_percentage', action='store_true')
    parser.add_argument('--no-flood_image_percentage', dest='flood_image_percentage', action='store_false')
    parser.add_argument('--single_region_distribution', dest='single_region_distribution', action='store_true')
    parser.add_argument('--no-single_region_distribution', dest='single_region_distribution', action='store_false')
    parser.add_argument('--single_fsbr', dest='single_fsbr', action='store_true')
    parser.add_argument('--no-single_fsbr', dest='single_fsbr', action='store_false')
    parser.add_argument('--single_fibr', dest='single_fibr', action='store_true')
    parser.add_argument('--no-fibr', dest='single_fibr', action='store_false')
    # [SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST, SPLIT_BOLIVIA, SPLIT_WEAK]
    parser.add_argument('--splits_to_show', dest='splits_to_show', nargs='+', type=str, default=[SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST, SPLIT_BOLIVIA])

    parser.set_defaults(include_analysis=False, show=True, save=True, with_figure_title=False,
                        single_flood_no_flood=True, flood_image_percentage=True, single_region_distribution=True,
                        single_fsbr=True, single_fibr=True)
    parser.set_defaults()
    args = parser.parse_args()
    plt.rcParams.update(args.rc_params)
    print('Font Sizes:', plt.rcParams['font.size'], plt.rcParams['legend.fontsize'], plt.rcParams['legend.title_fontsize'])
    print('Font Properties:', plt.rcParams['font.family'])
    print('Font Weight:', plt.rcParams['font.weight'])

    SPLITS_TO_SHOW = args.splits_to_show
    fig, axes, ld = create_figure_with_subplots('Sen1Floods11-per-Split Distribution', 5, len(SPLITS_TO_SHOW))
    region_colors = {}
    flood_pixels_by_region, flood_images = [], []
    for i, split in enumerate(SPLITS_TO_SHOW):
        source = Sen1Floods11DataDingsDatasource(args.data_folder, (TYPE_LABEL if split != SPLIT_WEAK else
                                                                    TYPE_S2_WEAK_LABEL),
                                                 split, as_array=False, in_memory=False)
        dataset, meta = source(None, None)
        region_colors, flood_pixels_by_region_df, flood_images_df = plot_set_distribution(split, axes[:, i], dataset,
                                                                                          meta, region_colors, args)
        if split != SPLIT_WEAK:
            flood_pixels_by_region.append(flood_pixels_by_region_df)
            flood_images.append(flood_images_df)
    save_and_show(fig, path.join(args.plot_folder, 'dataset_split_dist'), args.show, ld, args.save)
    fig, axes, ld = create_figure_with_subplots('Sen1Floods11-Hand-Labeled Distribution', 1, 5)

    fpbr = pd.concat(flood_pixels_by_region).sort_values(['count', 'region'])
    fi_df = pd.concat(flood_images).sort_values(['region', 'is_no_data_only', 'is_dry_only', 'is_no_flood'])
    plot_distribution('_'.join(SPLITS_TO_SHOW), fpbr, fi_df, axes[0], region_colors, create_per_region_df(fpbr), args)
    save_and_show(fig, path.join(args.plot_folder, 'dataset_dist'), args.show, ld, args.save)
