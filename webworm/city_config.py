# city_config.py
CITY_CONFIG = {
    'data_sources': {
        'baidu_baike': 'https://baike.baidu.com/item/',
        'tianqi': 'http://www.tianqi.com/',
        'dianping': 'https://www.dianping.com/search/keyword/',
        'gaode_map': 'https://ditu.amap.com/search?query='
    },
    'crawler_settings': {
        'timeout': 10,
        'delay': 2,
        'max_retries': 3,
        'headless': False
    },
    'output_fields': [
        'city_name', 'population', 'area', 'gdp', 'administrative_level',
        'air_quality', 'pm25', 'green_coverage', 'water_quality',
        'history_years', 'cultural_sites', 'universities', 'museums',
        'subway_lines', 'airport_level', 'railway_stations',
        'climate_type', 'avg_temperature', 'rainfall',
        'local_cuisine', 'famous_dishes', 'restaurant_count'
    ]
}

# 352个城市列表
CITIES_352 = [
    '北京', '上海', '天津', '重庆', '石家庄', '唐山', '秦皇岛', '邯郸',
    '邢台', '保定', '张家口', '承德', '沧州', '廊坊', '衡水', '太原',
    '大同', '阳泉', '长治', '晋城', '朔州', '晋中', '运城', '忻州',
    '临汾', '吕梁', '呼和浩特', '包头', '乌海', '赤峰', '通辽', '鄂尔多斯',
    # ... 添加完整的352个城市列表
    '广州', '深圳', '珠海', '汕头', '佛山', '韶关', '湛江', '肇庆'
]