from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

from deoldify.visualize import *
plt.style.use('dark_background')
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

colorizer = get_image_colorizer(artistic=False)

result_path = None
render_factor=35


# python VideoColorizer_test.py
# zip -r results.zip results

print("start")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/001'
# result = '/root/autodl-tmp/DeOldify-master/test/results/001'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("001 finished")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/002'
# result = '/root/autodl-tmp/DeOldify-master/test/results/002'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("002 finished")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/003'
# result = '/root/autodl-tmp/DeOldify-master/test/results/003'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("003 finished")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/004'
# result = '/root/autodl-tmp/DeOldify-master/test/results/004'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("004 finished")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/005'
# result = '/root/autodl-tmp/DeOldify-master/test/results/005'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("005 finished")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/006'
# result = '/root/autodl-tmp/DeOldify-master/test/results/006'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("006 finished")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/007'
# result = '/root/autodl-tmp/DeOldify-master/test/results/007'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("007 finished")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/008'
# result = '/root/autodl-tmp/DeOldify-master/test/results/008'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("008 finished")

# source = '/root/autodl-tmp/DeOldify-master/test/bwframes/009'
# result = '/root/autodl-tmp/DeOldify-master/test/results/009'
# files = os.listdir(source)
# for i in files:
#     source_path = os.path.join(source, i)
#     result_new = os.path.join(result, i)
#     result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
# print("009 finished")

source = '/root/autodl-tmp/DeOldify-master/test/bwframes/010'
result = '/root/autodl-tmp/DeOldify-master/test/results/010'
files = os.listdir(source)
for i in files:
    source_path = os.path.join(source, i)
    result_new = os.path.join(result, i)
    result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
print("010 finished")

source = '/root/autodl-tmp/DeOldify-master/test/bwframes/011'
result = '/root/autodl-tmp/DeOldify-master/test/results/011'
files = os.listdir(source)
for i in files:
    source_path = os.path.join(source, i)
    result_new = os.path.join(result, i)
    result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
print("011 finished")

source = '/root/autodl-tmp/DeOldify-master/test/bwframes/012'
result = '/root/autodl-tmp/DeOldify-master/test/results/012'
files = os.listdir(source)
for i in files:
    source_path = os.path.join(source, i)
    result_new = os.path.join(result, i)
    result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
print("012 finished")

source = '/root/autodl-tmp/DeOldify-master/test/bwframes/013'
result = '/root/autodl-tmp/DeOldify-master/test/results/013'
files = os.listdir(source)
for i in files:
    source_path = os.path.join(source, i)
    result_new = os.path.join(result, i)
    result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
print("013 finished")

source = '/root/autodl-tmp/DeOldify-master/test/bwframes/014'
result = '/root/autodl-tmp/DeOldify-master/test/results/014'
files = os.listdir(source)
for i in files:
    source_path = os.path.join(source, i)
    result_new = os.path.join(result, i)
    result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
print("014 finished")

source = '/root/autodl-tmp/DeOldify-master/test/bwframes/015'
result = '/root/autodl-tmp/DeOldify-master/test/results/015'
files = os.listdir(source)
for i in files:
    source_path = os.path.join(source, i)
    result_new = os.path.join(result, i)
    result_path = colorizer.plot_transformed_image(path=source_path, results_dir=result_new, render_factor=render_factor, compare=True)
print("finished")
