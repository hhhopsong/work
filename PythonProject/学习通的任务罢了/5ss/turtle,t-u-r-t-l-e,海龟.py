
words = {'turtle': '海龟', 'amateur': '业余的; 业余爱好者', 'cape': '海角; 披肩,斗篷', 'comedy': '喜剧', 'estate': '房地产'}
count = 0
for eng, chi in words.items():
    count += 1
    print(f'{count}.{eng},{"-".join(list(eng))},{chi}')

