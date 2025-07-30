import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from pypinyin import lazy_pinyin

HEADERS = {"User-Agent": "Mozilla/5.0"}

def normalize_text(text):
    # 去掉所有不可见空白字符（包括全角/半角空格、换行、制表等）
    return re.sub(r'\s+', '', text.strip())

def get_first_by_keys(dictionary, keys):
    # 宽松匹配字段名
    for k, v in dictionary.items():
        for cand in keys:
            if normalize_text(cand) == normalize_text(k):
                return v
    return ""

def get_baike_info(city):
    url = f"https://baike.baidu.com/item/{city}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    html = resp.text
    
    # 检查是否为多义项分支页面（多义词页面）
    # -------- 多义词页面处理 ----------
    # 多义词页面鲁棒检测——查找所有含polysemantList的标签
    # polysemant = (soup.find(class_=re.compile("polysemantList")) or
    #               soup.find(string=re.compile("多义词")) or
    #               soup.find(string=re.compile("下列义项")) )
    

    # 直接找所有a标签，遍历找“市”义项
    if "本词条是一个多义词，请在下列义项中选择浏览" in html:
        # print("本词条是一个多义词，请在下列义项中选择浏览")

        candidate_url = None
        for a in soup.find_all("a", href=True):
            name = a.get_text()
            # 判断机制仍以“市”或“地级市”为主，适配更多义项变种
            if ("地级市" in name) :
                # print("匹配到 candidate_url:", name)
                candidate_url = a['href']
                # 优先最精确地级市
                if "地级市" in name:
                    break
        if candidate_url:
            real_url = "https://baike.baidu.com" + candidate_url
            resp = requests.get(real_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
    result={}
    # 1. info卡片区（dt/dd，通常都在这里）
    info_dict = {}
    for dl in soup.find_all('dl'):
        dts = dl.find_all('dt')
        dds = dl.find_all('dd')
        for dt, dd in zip(dts, dds):
            dt_text = normalize_text(dt.get_text(strip=True))
            dd_text = dd.get_text(strip=True)
            if len(dt_text) < 12:
                info_dict[dt_text] = dd_text
    # 2. 字段映射查卡片（宽松匹配）
    fieldmap = {
        'population': ['常住人口','人口数量','人口'],
        'area': ['面积','总面积'],
        'gdp': ['地区生产总值','GDP','生产总值'],
        'administrative_level': ['行政区类别'],
        'airport': ['机场'],
        'train_station' : ['火车站'],
        'climate_type': ['气候条件'],
        'travel resort': ['著名景点'],
        # 其它: 自己可添加
    }
    for field, candidates in fieldmap.items():
        value = get_first_by_keys(info_dict, candidates)
        result[field] = value
    # 4. 正则等兜底法查正文（如果卡片查不到，正文扫一遍）
    text = soup.get_text(separator='\n')
    if not result['population']:
        pop = re.search(r"(常住人口|人口数量|人口)[：:，]?\s*([\d\.]+[万亿]?)(人|$)", text)
        if pop:
            result['population'] = pop.group(2)
    if not result['area']:
        area = re.search(r"(总面积|面积)[：:，]?\s*([\d,\.]+)\s*(km²|平方千米|平方公里|KM2|KM²)", text, flags=re.I)
        if area:
            result['area'] = area.group(2) + area.group(3)
    if not result['gdp']:
        gdp = re.search(r"(GDP|地区生产总值|生产总值)[：:，]?\s*([\d\.,]+[万亿]?元?)", text)
        if gdp:
            result['gdp'] = gdp.group(2)
    if not result['airport']:
        air = re.search(r"机\s*场[：:，]?\s*([^\n，。,;； ]+)", text)
        if air:
            result['airport'] = air.group(1)
    if not result['train_station']:
        tra = re.search(r"火\s*车\s*站[：:，]?\s*([^\n，。,;； ]+)", text)
        if tra:
            result['train_station'] = tra.group(1)
    # 其它字段依此类推
    return result

def get_tianqi_info(city):
    city_pinyin = ''.join(lazy_pinyin(city))
    url = f"https://www.tianqi.com/{city_pinyin}/"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    result = {}
    aqi = soup.find("span", class_="aqi")
    result["air_quality"] = aqi.get_text(strip=True) if aqi else ""
    return result

def crawl_city(city):
    baike_data = get_baike_info(city)
    tianqi_data = get_tianqi_info(city)
    data = {**baike_data, **tianqi_data}
    data['city'] = city
    return data

if __name__ == "__main__":
    cities = ["北京市", "天津市", "石家庄市", "唐山市", "秦皇岛市", "邯郸市", "邢台市", "保定市", "张家口市", "承德市", "沧州市", "廊坊市", "衡水市",
"太原市", "大同市", "阳泉市", "长治市", "晋城市", "朔州市", "晋中市", "运城市", "忻州市", "临汾市", "吕梁市",
"呼和浩特市", "包头市", "乌海市", "赤峰市", "通辽市", "鄂尔多斯市", "呼伦贝尔市", "巴彦淖尔市", "乌兰察布市", "兴安盟", "锡林郭勒盟", "阿拉善盟",
"沈阳市", "大连市", "鞍山市", "抚顺市", "本溪市", "丹东市", "锦州市", "营口市", "阜新市", "辽阳市", "盘锦市", "铁岭市", "朝阳市", "葫芦岛市",
"长春市", "吉林市", "四平市", "辽源市", "通化市", "白山市", "松原市", "白城市", "延边朝鲜族自治州",
"哈尔滨市", "齐齐哈尔市", "鸡西市", "鹤岗市", "双鸭山市", "大庆市", "伊春市", "佳木斯市", "七台河市", "牡丹江市", "黑河市", "绥化市", "大兴安岭地区",
"上海市",
"南京市", "无锡市", "徐州市", "常州市", "苏州市", "南通市", "连云港市", "淮安市", "盐城市", "扬州市", "镇江市", "泰州市", "宿迁市",
"杭州市", "宁波市", "温州市", "嘉兴市", "湖州市", "绍兴市", "金华市", "衢州市", "舟山市", "台州市", "丽水市",
"合肥市", "芜湖市", "蚌埠市", "淮南市", "马鞍山市", "淮北市", "铜陵市", "安庆市", "黄山市", "滁州市", "阜阳市", "宿州市", "六安市", "亳州市", "池州市", "宣城市",
"福州市", "厦门市", "莆田市", "三明市", "泉州市", "漳州市", "南平市", "龙岩市", "宁德市",
"南昌市", "景德镇市", "萍乡市", "九江市", "新余市", "鹰潭市", "赣州市", "吉安市", "宜春市", "抚州市", "上饶市",
"济南市", "青岛市", "淄博市", "枣庄市", "东营市", "烟台市", "潍坊市", "济宁市", "泰安市", "威海市", "日照市", "临沂市", "德州市", "聊城市", "滨州市", "菏泽市",
"郑州市", "开封市", "洛阳市", "平顶山市", "安阳市", "鹤壁市", "新乡市", "焦作市", "濮阳市", "许昌市", "漯河市", "三门峡市", "南阳市", "商丘市", "信阳市", "周口市", "驻马店市", "济源市",
"武汉市", "黄石市", "十堰市", "宜昌市", "襄阳市", "鄂州市", "荆门市", "孝感市", "荆州市", "黄冈市", "咸宁市", "随州市", "恩施土家族苗族自治州",
"长沙市", "株洲市", "湘潭市", "衡阳市", "邵阳市", "岳阳市", "常德市", "张家界市", "益阳市", "郴州市", "永州市", "怀化市", "娄底市", "湘西土家族苗族自治州",
"广州市", "韶关市", "深圳市", "珠海市", "汕头市", "佛山市", "江门市", "湛江市", "茂名市", "肇庆市", "惠州市", "梅州市", "汕尾市", "河源市", "阳江市", "清远市", "东莞市", "中山市", "潮州市", "揭阳市", "云浮市",
"南宁市", "柳州市", "桂林市", "梧州市", "北海市", "防城港市", "钦州市", "贵港市", "玉林市", "百色市", "贺州市", "河池市", "来宾市", "崇左市",
"海口市", "三亚市", "三沙市", "儋州市",
"重庆市",
"成都市", "自贡市", "攀枝花市", "泸州市", "德阳市", "绵阳市", "广元市", "遂宁市", "内江市", "乐山市", "南充市", "眉山市", "宜宾市", "广安市", "达州市", "雅安市", "巴中市", "资阳市", "阿坝藏族羌族自治州", "甘孜藏族自治州", "凉山彝族自治州",
"贵阳市", "六盘水市", "遵义市", "安顺市", "毕节市", "铜仁市", "黔西南布依族苗族自治州", "黔东南苗族侗族自治州", "黔南布依族苗族自治州",
"昆明市", "曲靖市", "玉溪市", "保山市", "昭通市", "丽江市", "普洱市", "临沧市", "楚雄彝族自治州", "红河哈尼族彝族自治州", "文山壮族苗族自治州", "西双版纳傣族自治州", "大理白族自治州", "德宏傣族景颇族自治州", "怒江傈僳族自治州", "迪庆藏族自治州",
"拉萨市", "日喀则市", "昌都市", "林芝市", "山南市", "那曲市", "阿里地区",
"西安市", "铜川市", "宝鸡市", "咸阳市", "渭南市", "延安市", "汉中市", "榆林市", "安康市", "商洛市",
"兰州市", "嘉峪关市", "金昌市", "白银市", "天水市", "武威市", "张掖市", "平凉市", "酒泉市", "庆阳市", "定西市", "陇南市", "临夏回族自治州", "甘南藏族自治州",
"西宁市", "海东市", "海北藏族自治州", "黄南藏族自治州", "海南藏族自治州", "果洛藏族自治州", "玉树藏族自治州", "海西蒙古族藏族自治州",
"银川市", "石嘴山市", "吴忠市", "固原市", "中卫市",
"乌鲁木齐市", "克拉玛依市", "吐鲁番市", "哈密市", "昌吉回族自治州", "博尔塔拉蒙古自治州", "巴音郭楞蒙古自治州", "阿克苏地区", "克孜勒苏柯尔克孜自治州", "喀什地区", "和田地区", "伊犁哈萨克自治州", "塔城地区", "阿勒泰地区"]
    #烟台市 厦门市
    results = []
    for city in cities:
        print(f"提取: {city}")
        res = crawl_city(city)
        print(res)
        results.append(res)
    df = pd.DataFrame(results, columns=[
        "population", "area", "gdp", "administrative_level", "airport", "train_station","climate_type", "travel resort", "city"
    ])
    df.to_csv("中国城市综合指标.csv", index=False, encoding="utf-8-sig")
    print("采集完成！")