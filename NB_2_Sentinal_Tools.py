#!/usr/bin/env python
# coding: utf-8

# ## <span style="color: #4682B4;">S5P NO2 Toolkit</span>
# ### <span style="color: #DC143C;">Streamlined Python functions for advanced satellite data management.</span>

# In[3]:


import sys
import subprocess
import pkg_resources
import importlib
# List of libraries to check/install/upgrade
libraries = [
    'pandas',
    'geopandas',
    'netCDF4',
    'numpy',
    'matplotlib',
    'requests',
    'xmltodict',
    'shapely'
]
# Libraries that aren't directly installable via pip
non_installable_libraries = [
    'calendar',
    'datetime',
    're',
    'socket',
    'subprocess',
    'mpl_toolkits.axes_grid1',
    'urllib'
]
# Function to install a package
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"🎉 Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing {package}: {e}")
# Function to upgrade a package
def upgrade_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
        print(f"⬆️ Successfully upgraded {package}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error upgrading {package}: {e}")
# Function to check if a package is installed and install/upgrade it
def check_and_install(library):
    if library in non_installable_libraries:
        print(f"🔹 {library} is part of the Python standard library or not installable via pip.")
        return
    try:
        pkg_resources.get_distribution(library)
        print(f"✅ {library} is already installed.")
        upgrade_package(library)
    except pkg_resources.DistributionNotFound:
        print(f"🚀 {library} is not installed. Installing now...")
        install_package(library)
# Check and install/upgrade packages
for library in libraries:
    check_and_install(library)
# Import all the necessary libraries
def import_libraries():
    globals().update(locals())
    import pandas as pd
    import geopandas as gpd
    from os import listdir, rename, path, remove, mkdir
    from os.path import isfile, join, getsize, exists
    from netCDF4 import Dataset
    import time
    import numpy as np
    import calendar
    import datetime as dt
    import re
    from socket import timeout
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    import urllib
    import requests, json
    from requests.auth import HTTPBasicAuth
    import xmltodict
    from shapely import wkt
# Try to import all libraries and handle any potential import errors
try:
    import_libraries()
    print("🌟 All libraries are imported successfully!")
except ImportError as e:
    print(f"❌ Error importing libraries: {e}")
print("✅ All libraries are checked, installed, upgraded, and imported successfully!")


# In[ ]:


def create_project(project_name='default'):
    """    
    Description
    -----------
    This function checks if the specified subfolder exists. If not, it attempts to create the subfolder.
    If the subfolder is successfully created, it returns the subfolder's name. Otherwise, it returns an
    empty string indicating failure. If the subfolder already exists, it returns the subfolder's name.
    """
    # Check if the directory already exists
    if not os.path.exists(project_name):
        try:
            # Attempt to create the directory
            os.mkdir(project_name)
        except OSError as e:
            # Handle any errors during the directory creation
            print(f"❌ Creation of the directory {project_name} failed: {e}")
            return ''
        else:
            # Successfully created the directory
            print(f"🎉 Successfully created the directory {project_name}.")
            return project_name
    else:
        # Directory already exists
        print(f"ℹ️ Directory {project_name} already exists.")
        return project_name


# In[ ]:


def get_place_boundingbox(place_gdf, buffer_distance):
    """
    Determine the bounding box for a given GeoDataFrame representing a place.
    Parameters
    ----------
    place_gdf : GeoDataFrame
        A GeoDataFrame containing the geographic data of the place. This should be a Level 0 polygon from GADM.
    buffer_distance : int
        The distance in miles to extend the boundaries of the place.
    Returns
    -------
    GeoDataFrame
        A new GeoDataFrame containing the bounding box around the place.
    Description
    -----------
    This function takes the geometric shape of a place, expands its boundaries by a specified distance,
    and returns a rectangle that fully contains the expanded shape.
    """
    # Create the bounding box with the specified buffer
    expanded_geometry = place_gdf['geometry'].buffer(buffer_distance).envelope
    bounding_box_gdf = gpd.GeoDataFrame(geometry=expanded_geometry, crs=place_gdf.crs).reset_index()
    return bounding_box_gdf


# In[7]:


def filter_swath_set(swath_set_gdf, place_gdf):
    """
    Filter swaths based on the place constraint.
    Parameters
    ----------
    swath_set_gdf : GeoDataFrame
        GeoDataFrame containing swath geometries.
    place_gdf : GeoDataFrame
        GeoDataFrame of the place to filter swaths for.
    Returns
    -------
    GeoDataFrame
        Subset of swath_set_gdf containing geometries that cover place_gdf.
    """
    filtered_gdf = gpd.sjoin(swath_set_gdf, place_gdf, how='right', op='contains').reset_index()
    filtered_gdf = filtered_gdf.drop(columns=['level_0','index_left','index'])
    return filtered_gdf


# In[8]:


def geometry_to_wkt(place_gdf):
    """
    Convert GeoDataFrame geometry to Well-Known Text (WKT) format.
    Description
    -----------
    This function converts the geometry of a given place GeoDataFrame into a Well-Known Text (WKT) format.
    This format is required for the Sentinel 5P Data Access hub to constrain the polygon filter, allowing
    for the retrieval of a smaller number of satellite image swaths.
    """
    # Get the geometry's convex hull and simplify it
    simplified_geometry = place_gdf.reset_index()['geometry'].convex_hull.simplify(tolerance=0.05)
    # Convert the simplified geometry to WKT
    wkt_string = wkt.dumps(simplified_geometry[0])
    return wkt_string


# In[9]:


def date_from_week(week_string='2019-W01'):
    """
    Convert a week string to a datetime object.
    """
    return dt.datetime.strptime(week_string + '-1', "%Y-W%W-%w")


# In[10]:


def add_days(start, num_days=1):
    """
    Add a number of days to a start date and return both dates.
    """
    end = start + dt.timedelta(days=num_days)
    return [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]


# In[12]:


def nc_to_df(ncfile):
    """
    Convert TROPOMI NO2 NetCDF file to DataFrame.
    """
    try:
        file = Dataset(ncfile, 'r')
    except OSError:
        print('Cannot open', ncfile)
        return pd.DataFrame()
    if 'NO2___' not in ncfile or 'S5P' not in ncfile:
        raise NameError('Not a TROPOMI NO2 file name.')
    grp = 'PRODUCT'
    lat = file[grp].variables['latitude'][0][:][:]
    lon = file[grp].variables['longitude'][0][:][:]
    data = file[grp].variables['nitrogendioxide_tropospheric_column']
    fv = data._FillValue
    scan_time = file[grp].variables['time_utc']
    timestamps = [dt.datetime.strptime(t.split('.')[0], '%Y-%m-%dT%H:%M:%S').timestamp() for t in scan_time[0]]
    df = pd.DataFrame({
        'UnixTimestamp': np.repeat(timestamps, lat.shape[1]),
        'DateTime': pd.to_datetime(np.repeat(timestamps, lat.shape[1]), unit='s')
    })
    df[['Date', 'Time']] = df['DateTime'].astype(str).str.split(' ', expand=True)
    for var in file[grp].variables.keys():
        sds = file[grp].variables[var]
        if len(sds.shape) == 3:
            scale = sds.scale_factor if 'qa' in var else 1.0
            data = np.where(sds[:].ravel() == fv, np.nan, sds[:].ravel() * scale)
            df[var] = data
    return df


# In[13]:


def polygon_filter(input_df, filter_gdf):
    """
    Remove records from NO2 DataFrame outside filter polygons.
    """
    print("Ensure you've created the spatial index for filter_gdf with filter_gdf.sindex.")
    tic = time.perf_counter()
    # Convert input_df to GeoDataFrame with same CRS as filter_gdf
    gdf = gpd.GeoDataFrame(
        input_df, 
        geometry=gpd.points_from_xy(input_df.longitude, input_df.latitude),
        crs=filter_gdf.crs
    )
    print(f"Original NO2 DataFrame length: {len(gdf)}")
    # Perform spatial join to filter data
    filtered_gdf = gpd.sjoin(gdf, filter_gdf, how='inner', op='intersects')
    filtered_gdf = filtered_gdf.drop(columns=['index_right'])
    print(f"Filtered NO2 GeoDataFrame length: {len(filtered_gdf)}")
    toc = time.perf_counter()
    print(f"Processed NO2 DataFrame sjoin in {str((toc - tic) / 60)} minutes")
    return filtered_gdf


# In[14]:


def get_filename_from_cd(cd):
    """
    Extract filename from content-disposition (cd) header.
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if not fname:
        return None
    return fname[0]


# In[1]:


def download_nc_file(url, auth, savedir, logging=False, refresh=False, chunk_size=1024*1024):
    """
    Downloads NetCDF files from a URL. Takes URL, user auth, save directory, logging, refresh, and chunk size as parameters. Returns the filename.
    """
    user, password = auth['user'], auth['password']
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    }
    try:
        tic = time.perf_counter()
        response = requests.get(url, auth=(user, password), stream=True, headers=headers)
        content_disposition = response.headers.get('content-disposition')
        filename = content_disposition.split('filename=')[-1].replace('"', '') if content_disposition else 'downloaded_file.nc'
        file_path = os.path.join(savedir, filename)
        if os.path.exists(file_path) and not refresh:
            if os.path.getsize(file_path) > 0:
                return filename
        with open(file_path, 'wb') as f:
            for data in response.iter_content(chunk_size=chunk_size):
                f.write(data)
        if logging:
            with open(os.path.join(savedir, 'nc.log'), 'a+') as l:
                l.seek(0)
                if l.read(100):
                    l.write("\n")
                l.write(filename)
        toc = time.perf_counter()
        print(f'Success: Saved {filename} to {savedir}.')
        print(f'Download time: {toc-tic:.2f} seconds')
        delay = np.random.choice([7, 4, 6, 2, 10, 15, 19, 23])
        print(f'Delaying for {delay} seconds...')
        time.sleep(delay)
        return filename
    except Exception as e:
        print('Something went wrong:', e)
        return None


# In[2]:


def harpconvert(input_filename, input_dir, output_dir):
    """
    Converts TROPOMI NO2 NetCDF to HDF5 (L3 Analysis).
    Args:
        input_filename (str): Name of the input NetCDF file.
        input_dir (str): Directory of the input file.
        output_dir (str): Directory to save the HDF5 file.
    Returns:
        dict: Contains filename, filesize, elapsed time, stdout, stderr.
    """
    tic = time.perf_counter()
    output_filename = input_filename.replace('.nc', '.h5')
    input_path = os.path.join(input_dir, input_filename)
    output_path = os.path.join(output_dir, output_filename)
    cmd = (
        f"harpconvert --format hdf5 --hdf5-compression 9 "
        f"-a 'tropospheric_NO2_column_number_density_validity>50;derive(datetime_stop {{time}})' "
        f"{input_path} {output_path}"
    )
    process = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    elapsed_time = time.perf_counter() - tic
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file {output_filename} was not created.")
    checksum = subprocess.check_output(['sha256sum', output_path]).split()[0].decode()
    status_dict = {
        'input_filename': input_filename,
        'output_filesize': f"{os.path.getsize(output_path):,} bytes",
        'elapsed_time': f"{elapsed_time:.2f} seconds",
        'stdout': stdout.decode(),
        'stderr': stderr.decode(),
        'checksum': checksum,
    }
    return status_dict


# In[3]:


def batch_assemble_filtered_pickles(filtered_dir):
    """
    Assembles DataFrames from pickle files in a directory.
    Args:
        filtered_dir (str): Directory containing the pickle files.
    Returns:
        DataFrame: Concatenated DataFrame of all pickle files.
    """
    tic = time.perf_counter()
    pickle_files = [f for f in os.listdir(filtered_dir) if os.path.isfile(os.path.join(filtered_dir, f))]
    full_df = pd.DataFrame()
    for pickle_file in pickle_files:
        print(pickle_file)
        df = pd.read_pickle(os.path.join(filtered_dir, pickle_file))
        full_df = pd.concat([df, full_df], axis=0)
    elapsed_time = (time.perf_counter() - tic) / 60
    print(f'Assembly time: {elapsed_time:.2f} minutes')
    output_filename = os.path.join(filtered_dir, 'assembled_dataframe.pkl')
    full_df.to_pickle(output_filename)
    print(f'Saved assembled DataFrame to {output_filename}')
    return full_df


# In[4]:


def plot_maps(iso3, filter_gdf, filelist, colormap, sensing_date):
    """
    Plots TROPOMI NO2 data on maps.
    Args:
        iso3 (str): 3-letter ISO country code.
        filter_gdf (GeoDataFrame): Filtered GeoDataFrame.
        filelist (list): List of pickle files.
        colormap (str): Colormap for the plot.
        sensing_date (str): Date of sensing.
    Returns:
        Matplotlib figure object.
    """
    crs = filter_gdf.crs
    country_gdf = filter_gdf[filter_gdf['iso3'] == iso3]
    country_name = country_gdf['name'].unique()[0]
    gdf_sjoin_list = []
    for file in filelist:
        gdf_sjoin = pd.read_pickle(file)
        gdf_sjoin = gdf_sjoin.set_geometry('geometry').to_crs(crs)
        gdf_countries_sjoin = gpd.sjoin(gdf_sjoin, country_gdf, how='inner', op='intersects')
        if not gdf_countries_sjoin.empty:
            gdf_sjoin_list.append(gdf_countries_sjoin)
    print(f'Using {len(gdf_sjoin_list)} swaths.')
    def get_column_range(gdf_list, column):
        return min(gdf[column].min() for gdf in gdf_list), max(gdf[column].max() for gdf in gdf_list)
    vmin_qa, vmax_qa = get_column_range(gdf_sjoin_list, 'qa_value')
    vmin_no2, vmax_no2 = get_column_range(gdf_sjoin_list, 'nitrogendioxide_tropospheric_column')
    def plot_with_colorbar(ax, gdf_list, column, vmin, vmax, title):
        for gdf in gdf_list:
            gdf.plot(ax=ax, column=column, cmap=plt.get_cmap(colormap), vmin=vmin, vmax=vmax, alpha=0.9)
        country_gdf.plot(ax=ax, color='None', edgecolor='black', alpha=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        ax.get_figure().colorbar(sm, cax=cax)
        ax.set_title(title)
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), constrained_layout=True, sharex=True, sharey=True)
    plot_with_colorbar(axs[0], gdf_sjoin_list, 'qa_value', vmin_qa, vmax_qa, f'Tropospheric NO2, QA Value ({country_name}, {sensing_date})')
    plot_with_colorbar(axs[1], gdf_sjoin_list, 'nitrogendioxide_tropospheric_column_precision_kernel', vmin_no2, vmax_no2, f'Tropospheric NO2, Tropospheric Column, moles/m² ({country_name}, {sensing_date})')
    return fig


# In[5]:


def sentinel_api_query(query_dict, silentmode=False):
    """
    Queries Sentinel-5P data and returns results as a GeoDataFrame.
    Args:
        query_dict (dict): API query variables.
        silentmode (bool): Suppress print statements if True.
    Returns:
        GeoDataFrame: GeoDataFrame containing the query results.
    """
    delay = np.random.choice([7, 4, 6, 2, 10, 15, 19, 23])
    if not silentmode:
        print(f'Delaying for {delay} seconds...')
    time.sleep(delay)
    # Unpack query_dict
    polygon = query_dict['polygon']
    startDate = query_dict['startDate']
    endDate = query_dict['endDate']
    platformName = query_dict['platformName']
    productType = query_dict['productType']
    processingLevel = query_dict['processingLevel']
    processingMode = query_dict['processingMode']
    dhus_url = query_dict['dhus_url']
    startPage = query_dict['startPage']
    numRows = query_dict['numRows']
    username = query_dict['username']
    password = query_dict['password']
    # Construct query string for API
    query = (
        f'( footprint:"Intersects({polygon})") AND '
        f'( beginPosition:[{startDate}T00:00:00.000Z TO {endDate}T23:59:59.999Z] AND '
        f'endPosition:[{startDate}T00:00:00.000Z TO {endDate}T23:59:59.999Z] ) AND '
        f'( (platformname:{platformName} AND producttype:{productType} '
        f'AND processinglevel:{processingLevel} AND processingmode:{processingMode}))'
    )
    quoted = urllib.parse.quote_plus(query)
    if not silentmode:
        print(f'query: {query}')
        print(f'quoted: {quoted}')
    # Send query to API and get response
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) Chrome/39.0.2171.95 Safari/537.36'}
    response = requests.get(f'{dhus_url}dhus/search?q={quoted}&start={startPage}&rows={numRows}',
                            auth=HTTPBasicAuth(username, password), headers=headers)
    if not silentmode:
        print(f'headers: {response.headers}')
        print(f'cookie: {response.headers.get("Set-Cookie", "")}')
    # Convert XML response to dictionary
    my_dict = xmltodict.parse(response.text)
    results = int(my_dict['feed']['opensearch:totalResults'])
    if results > numRows and not silentmode:
        print(f'WARNING: Returned results {results} exceeds requested number of rows ({numRows}).')
    # Store dictionary items in Pandas DataFrame
    records = []
    for item in my_dict['feed']['entry']:
        gmldict = xmltodict.parse(item['str'][1]['#text'])
        crs = gmldict['gml:Polygon']['@srsName'].split('#')
        record = {
            'ingestiondate': item['date'][0]['#text'],
            'beginposition': item['date'][1]['#text'],
            'endposition': item['date'][2]['#text'],
            'orbitnumber': item['int']['#text'],
            'filename': item['str'][0]['#text'],
            'crs': f'epsg:{crs[1]}',
            'format': item['str'][2]['#text'],
            'identifier': item['str'][3]['#text'],
            'instrumentname': item['str'][4]['#text'],
            'instrumentshortname': item['str'][5]['#text'],
            'footprint': item['str'][6]['#text'],
            'mission': item['str'][7]['#text'],
            'platformname': item['str'][8]['#text'],
            'platformserialidentifier': item['str'][9]['#text'],
            'platformshortname': item['str'][10]['#text'],
            'processinglevel': item['str'][11]['#text'],
            'processingmode': item['str'][12]['#text'],
            'processingmodeabbreviation': item['str'][13]['#text'],
            'processorversion': item['str'][14]['#text'],
            'producttype': item['str'][15]['#text'],
            'producttypedescription': item['str'][16]['#text'],
            'revisionnumber': item['str'][17]['#text'],
            'size': item['str'][18]['#text'],
            'uuid': item['str'][19]['#text'],
            'downloadurl': item['link'][0]['@href']
        }
        records.append(record)
        if not silentmode:
            print(record)
    # Convert DataFrame to GeoDataFrame and return
    study_df = pd.DataFrame(records)
    study_df['geometry'] = study_df['footprint'].apply(wkt.loads)
    study_df['beginposition'] = pd.to_datetime(study_df['beginposition'].str.replace('T', ' '))
    study_df['endposition'] = pd.to_datetime(study_df['endposition'].str.replace('T', ' '))
    study_df['startdate'] = study_df['beginposition'].dt.strftime('%Y-%m-%d')
    study_df['enddate'] = study_df['endposition'].dt.strftime('%Y-%m-%d')
    study_gdf = gpd.GeoDataFrame(study_df, crs={'init': 'epsg:4326'}, geometry='geometry')
    return study_gdf


# In[6]:


def plot_color_gradients(gradient, cmap_category, cmap_list, save=False, output_dir='.'):
    """
    Plots color gradients for a given list of colormaps.
    Args:
        gradient (ndarray): Gradient to display colormaps.
        cmap_category (str): Category of the colormaps.
        cmap_list (list): List of colormap names.
        save (bool): Save the plot if True.
        output_dir (str): Directory to save the plots.
    """
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(cmap_list)))
    for i, cmap_name in enumerate(cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap_name))
        ax.text(-0.01, 0.5 * (2 * i + 1), cmap_name, va='center', ha='right', fontsize=10, transform=ax.transAxes)
    ax.set_axis_off()
    plt.title(cmap_category, loc='left', fontsize=12)
    if save:
        plt.savefig(f"{output_dir}/{cmap_category}.png")
    plt.show()
def show_colormap(filter_category=None, save=False, output_dir='.', resolution=256):
    """
    Displays available colormaps, with options to filter by category, save the plots, and set gradient resolution.
    Args:
        filter_category (str): Filter colormap categories (e.g., 'Sequential').
        save (bool): Save the plots if True.
        output_dir (str): Directory to save the plots.
        resolution (int): Resolution of the gradient.
    Returns:
        bool: True if successful.
    """
    cmaps = [
        ('Perceptually Uniform Sequential', ['viridis', 'plasma', 'inferno', 'magma', 'cividis']),
        ('Sequential', ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
        ('Sequential (2)', ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']),
        ('Diverging', ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
        ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
        ('Qualitative', ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']),
        ('Miscellaneous', ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])
    ]
    gradient = np.linspace(0, 1, resolution)
    gradient = np.vstack((gradient, gradient))
    for cmap_category, cmap_list in cmaps:
        if filter_category and cmap_category != filter_category:
            continue
        plot_color_gradients(gradient, cmap_category, cmap_list, save=save, output_dir=output_dir)
    return True


# In[7]:


def plot_color_gradients(gradient, cmap_category, cmap_list, save=False, output_dir='.', title=None, resolution=256):
    """
    Plots color gradients for a given list of colormaps with options to save and customize the plot.
    Args:
        gradient (ndarray): Gradient to display colormaps.
        cmap_category (str): Category of the colormaps.
        cmap_list (list): List of colormap names.
        save (bool): Save the plot if True.
        output_dir (str): Directory to save the plots.
        title (str): Custom title for the plot.
        resolution (int): Resolution of the gradient.
    """
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)
    axes[0].set_title(title if title else cmap_category + ' colormaps', fontsize=14)
    gradient = np.linspace(0, 1, resolution)
    gradient = np.vstack((gradient, gradient))
    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-.01, .5, name, va='center', ha='right', fontsize=10, transform=ax.transAxes)
    for ax in axes:
        ax.set_axis_off()
    if save:
        plt.savefig(f"{output_dir}/{cmap_category}.png")
    plt.show()

