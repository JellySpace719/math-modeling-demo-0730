# city_crawler.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import json
import logging
from datetime import datetime
import re
import numpy as np

# 导入配置和验证模块
from city_config import CITY_CONFIG, CITIES_352
from city_data_validator import CityDataValidator

class CityDataCrawler:
    def __init__(self, headless=False, timeout=10):
        """
        城市综合数据爬虫
        """
        self.config = CITY_CONFIG
        self.timeout = timeout
        self.setup_driver(headless)
        self.setup_logging()
        self.failed_cities = []
        self.success_count = 0
        self.total_count = 0
        self.validator = CityDataValidator()
        
        # 数据源配置
        self.data_sources = self.config['data_sources']
        
    def setup_driver(self, headless):
        """设置浏览器驱动"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, self.timeout)
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('city_crawler.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_city_data(self, city_name):
        """
        提取单个城市的综合数据
        """
        try:
            data = {
                'city_name': city_name,
                'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                # 城市规模
                'population': '',
                'area': '',
                'gdp': '',
                'administrative_level': '',
                # 环境环保
                'air_quality': '',
                'pm25': '',
                'green_coverage': '',
                'water_quality': '',
                # 人文底蕴
                'history_years': '',
                'cultural_sites': '',
                'universities': '',
                'museums': '',
                # 交通便利
                'subway_lines': '',
                'airport_level': '',
                'railway_stations': '',
                'highway_accessibility': '',
                # 气候
                'climate_type': '',
                'avg_temperature': '',
                'rainfall': '',
                'sunshine_hours': '',
                # 美食
                'local_cuisine': '',
                'famous_dishes': '',
                'food_rating': '',
                'restaurant_count': ''
            }
            
            # 从不同数据源获取数据
            self.get_city_basic_info(city_name, data)
            self.get_environmental_data(city_name, data)
            self.get_cultural_data(city_name, data)
            self.get_transportation_data(city_name, data)
            self.get_climate_data(city_name, data)
            self.get_food_data(city_name, data)
            
            self.success_count += 1
            self.logger.info(f"成功爬取: {city_name} ({self.success_count}/{self.total_count})")
            
            return data
            
        except Exception as e:
            self.logger.error(f"爬取失败 {city_name}: {str(e)}")
            self.failed_cities.append({'city': city_name, 'error': str(e)})
            return None
    
    def get_city_basic_info(self, city_name, data):
        """获取城市基本信息（规模数据）"""
        try:
            # 从百度百科获取基本信息
            url = f"{self.data_sources['baidu_baike']}{city_name}"
            self.driver.get(url)
            time.sleep(2)
            
            # 提取人口数据
            population_selectors = [
                "//dt[contains(text(), '人口')]/following-sibling::dd",
                "//span[contains(text(), '常住人口')]/parent::*/following-sibling::*",
                "//*[contains(text(), '万人')]"
            ]
            data['population'] = self.extract_by_selectors(population_selectors, r'(\d+\.?\d*)\s*万?人?')
            
            # 提取面积数据
            area_selectors = [
                "//dt[contains(text(), '面积')]/following-sibling::dd",
                "//span[contains(text(), '总面积')]/parent::*/following-sibling::*",
                "//*[contains(text(), '平方公里')]"
            ]
            data['area'] = self.extract_by_selectors(area_selectors, r'(\d+\.?\d*)\s*平方公里?')
            
            # 提取GDP数据
            gdp_selectors = [
                "//dt[contains(text(), 'GDP')]/following-sibling::dd",
                "//span[contains(text(), '生产总值')]/parent::*/following-sibling::*"
            ]
            data['gdp'] = self.extract_by_selectors(gdp_selectors, r'(\d+\.?\d*)\s*[万亿]?元?')
            
            # 行政级别
            admin_selectors = [
                "//dt[contains(text(), '行政区类别')]/following-sibling::dd",
                "//*[contains(text(), '地级市') or contains(text(), '副省级市') or contains(text(), '直辖市')]"
            ]
            data['administrative_level'] = self.extract_by_selectors(admin_selectors)
            
        except Exception as e:
            self.logger.warning(f"获取{city_name}基本信息失败: {e}")
    
    def get_environmental_data(self, city_name, data):
        """获取环境数据"""
        try:
            # 从天气网获取空气质量数据
            url = f"http://www.tianqi.com/{city_name}"
            self.driver.get(url)
            time.sleep(2)
            
            # 空气质量
            aqi_selectors = [
                "//span[contains(@class, 'aqi')]",
                "//*[contains(text(), 'AQI')]/following-sibling::*",
                "//div[contains(@class, 'air-quality')]"
            ]
            data['air_quality'] = self.extract_by_selectors(aqi_selectors, r'(\d+)')
            
            # PM2.5数据
            pm25_selectors = [
                "//span[contains(text(), 'PM2.5')]/following-sibling::span",
                "//*[contains(text(), 'PM2.5：')]/following-sibling::*"
            ]
            data['pm25'] = self.extract_by_selectors(pm25_selectors, r'(\d+)')
            
            # 从政府网站获取绿化覆盖率等数据
            self.get_gov_environmental_data(city_name, data)
            
        except Exception as e:
            self.logger.warning(f"获取{city_name}环境数据失败: {e}")
    
    def get_cultural_data(self, city_name, data):
        """获取人文底蕴数据"""
        try:
            # 从百度百科获取历史文化信息
            url = f"{self.data_sources['baidu_baike']}{city_name}"
            self.driver.get(url)
            time.sleep(2)
            
            # 建城历史
            history_selectors = [
                "//dt[contains(text(), '建立时间')]/following-sibling::dd",
                "//*[contains(text(), '建城于') or contains(text(), '始建于')]",
                "//*[contains(text(), '年历史')]"
            ]
            history_text = self.extract_by_selectors(history_selectors)
            if history_text:
                # 提取年份信息
                year_match = re.search(r'(\d{3,4})年?', history_text)
                if year_match:
                    current_year = datetime.now().year
                    history_years = current_year - int(year_match.group(1))
                    data['history_years'] = str(history_years)
            
            # 文物古迹数量
            cultural_selectors = [
                "//*[contains(text(), '全国重点文物保护单位')]",
                "//*[contains(text(), '文物古迹')]",
                "//dt[contains(text(), '名胜古迹')]/following-sibling::dd"
            ]
            data['cultural_sites'] = self.extract_by_selectors(cultural_selectors, r'(\d+)')
            
            # 高等院校数量
            university_selectors = [
                "//*[contains(text(), '高等院校') or contains(text(), '大学')]",
                "//dt[contains(text(), '高等学府')]/following-sibling::dd"
            ]
            data['universities'] = self.extract_by_selectors(university_selectors, r'(\d+)')
            
            # 博物馆数量
            museum_selectors = [
                "//*[contains(text(), '博物馆')]",
                "//dt[contains(text(), '文化场馆')]/following-sibling::dd"
            ]
            data['museums'] = self.extract_by_selectors(museum_selectors, r'(\d+)')
            
        except Exception as e:
            self.logger.warning(f"获取{city_name}文化数据失败: {e}")
    
    def get_transportation_data(self, city_name, data):
        """获取交通便利数据"""
        try:
            # 从高德地图获取交通信息
            url = f"{self.data_sources['gaode_map']}{city_name}地铁"
            self.driver.get(url)
            time.sleep(3)
            
            # 地铁线路数
            subway_selectors = [
                "//*[contains(text(), '号线')]",
                "//span[contains(@class, 'subway-line')]"
            ]
            subway_elements = self.driver.find_elements(By.XPATH, "//span[contains(text(), '号线')]")
            if subway_elements:
                data['subway_lines'] = str(len(subway_elements))
            
            # 从百度百科获取其他交通信息
            url = f"{self.data_sources['baidu_baike']}{city_name}"
            self.driver.get(url)
            time.sleep(2)
            
            # 机场等级
            airport_selectors = [
                "//*[contains(text(), '国际机场') or contains(text(), '机场')]",
                "//dt[contains(text(), '机场')]/following-sibling::dd"
            ]
            airport_text = self.extract_by_selectors(airport_selectors)
            if '国际' in airport_text:
                data['airport_level'] = '国际机场'
            elif '机场' in airport_text:
                data['airport_level'] = '国内机场'
            
            # 火车站数量
            station_selectors = [
                "//*[contains(text(), '火车站') or contains(text(), '高铁站')]",
                "//dt[contains(text(), '火车站')]/following-sibling::dd"
            ]
            data['railway_stations'] = self.extract_by_selectors(station_selectors, r'(\d+)')
            
            # 高速公路通达性
            highway_selectors = [
                "//*[contains(text(), '高速公路')]",
                "//dt[contains(text(), '交通')]/following-sibling::dd"
            ]
            highway_text = self.extract_by_selectors(highway_selectors)
            if highway_text:
                highway_count = len(re.findall(r'G\d+|S\d+', highway_text))
                data['highway_accessibility'] = str(highway_count) if highway_count > 0 else '良好'
            
        except Exception as e:
            self.logger.warning(f"获取{city_name}交通数据失败: {e}")
    
    def get_climate_data(self, city_name, data):
        """获取气候数据"""
        try:
            # 从天气网获取气候信息
            url = f"http://www.tianqi.com/{city_name}"
            self.driver.get(url)
            time.sleep(2)
            
            # 气候类型
            climate_selectors = [
                "//*[contains(text(), '亚热带') or contains(text(), '温带') or contains(text(), '热带')]",
                "//span[contains(@class, 'climate')]"
            ]
            data['climate_type'] = self.extract_by_selectors(climate_selectors)
            
            # 平均气温
            temp_selectors = [
                "//span[contains(@class, 'temperature')]",
                "//*[contains(text(), '平均气温')]"
            ]
            data['avg_temperature'] = self.extract_by_selectors(temp_selectors, r'(\d+\.?\d*)°?C?')
            
            # 年降雨量
            rain_selectors = [
                "//*[contains(text(), '降雨量') or contains(text(), '降水量')]",
                "//span[contains(@class, 'rainfall')]"
            ]
            data['rainfall'] = self.extract_by_selectors(rain_selectors, r'(\d+\.?\d*)\s*毫米?')
            
            # 日照时数
            sunshine_selectors = [
                "//*[contains(text(), '日照') or contains(text(), '光照')]",
                "//span[contains(@class, 'sunshine')]"
            ]
            data['sunshine_hours'] = self.extract_by_selectors(sunshine_selectors, r'(\d+\.?\d*)\s*小时?')
            
        except Exception as e:
            self.logger.warning(f"获取{city_name}气候数据失败: {e}")
    
    def get_food_data(self, city_name, data):
        """获取美食数据"""
        try:
            # 从大众点评获取美食信息
            url = f"{self.data_sources['dianping']}{city_name}美食"
            self.driver.get(url)
            time.sleep(3)
            
            # 特色菜系
            cuisine_selectors = [
                "//span[contains(@class, 'cuisine')]",
                "//*[contains(text(), '菜') and not(contains(text(), '青菜'))]"
            ]
            data['local_cuisine'] = self.extract_by_selectors(cuisine_selectors)
            
            # 餐厅数量统计
            restaurant_selectors = [
                "//span[contains(@class, 'count')]",
                "//*[contains(text(), '家餐厅')]"
            ]
            data['restaurant_count'] = self.extract_by_selectors(restaurant_selectors, r'(\d+)')
            
            # 美食评分
            rating_selectors = [
                "//span[contains(@class, 'rating')]",
                "//*[contains(@class, 'score')]"
            ]
            data['food_rating'] = self.extract_by_selectors(rating_selectors, r'(\d+\.?\d*)')
            
            # 从百度百科获取特色美食
            url = f"{self.data_sources['baidu_baike']}{city_name}特色美食"
            self.driver.get(url)
            time.sleep(2)
            
            # 特色菜品
            dish_selectors = [
                "//dt[contains(text(), '特色美食')]/following-sibling::dd",
                "//*[contains(text(), '著名小吃') or contains(text(), '特色菜')]"
            ]
            data['famous_dishes'] = self.extract_by_selectors(dish_selectors)
            
        except Exception as e:
            self.logger.warning(f"获取{city_name}美食数据失败: {e}")
    
    def get_gov_environmental_data(self, city_name, data):
        """从政府网站获取环境数据"""
        try:
            # 这里可以添加政府统计网站的爬取逻辑
            # 由于政府网站结构复杂，这里提供框架
            
            # 绿化覆盖率（从统计年鉴等获取）
            green_selectors = [
                "//*[contains(text(), '绿化覆盖率')]",
                "//*[contains(text(), '绿地率')]"
            ]
            data['green_coverage'] = self.extract_by_selectors(green_selectors, r'(\d+\.?\d*)%?')
            
            # 水质情况
            water_selectors = [
                "//*[contains(text(), '水质')]",
                "//*[contains(text(), '饮用水')]"
            ]
            water_text = self.extract_by_selectors(water_selectors)
            if '优' in water_text or 'I类' in water_text or 'II类' in water_text:
                data['water_quality'] = '优良'
            elif '良' in water_text or 'III类' in water_text:
                data['water_quality'] = '良好'
            else:
                data['water_quality'] = water_text
            
        except Exception as e:
            self.logger.warning(f"获取{city_name}政府环境数据失败: {e}")
    
    def extract_by_selectors(self, selectors, pattern=None):
        """通过多个选择器提取数据"""
        for selector in selectors:
            try:
                element = self.driver.find_element(By.XPATH, selector)
                text = element.text.strip()
                if text:
                    if pattern:
                        match = re.search(pattern, text)
                        if match:
                            return match.group(1)
                    else:
                        return text
            except:
                continue
        return ''
    
    def batch_crawl_cities(self, city_list, start_index=0):
        """
        批量爬取城市数据
        """
        self.total_count = len(city_list)
        results = []
        
        self.logger.info(f"开始爬取，总计 {self.total_count} 个城市")
        
        for i, city_name in enumerate(city_list):
            if i < start_index:
                continue
            
            self.logger.info(f"正在爬取: {city_name} ({i+1}/{self.total_count})")
            
            # 爬取数据
            data = self.extract_city_data(city_name)
            if data:
                results.append(data)
            
            # 保存进度
            if len(results) % 10 == 0 and results:
                self.save_progress(results, f'city_progress_{len(results)}.xlsx')
            
            # 延时避免被封
            time.sleep(self.config['crawler_settings']['delay'])
        
        # 保存最终结果
        self.save_results(results)
        
        return results
    
    def save_progress(self, data, filename):
        """保存进度"""
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False, encoding='utf-8')
        self.logger.info(f"进度已保存: {filename}")
    
    def save_results(self, data, filename='city_comprehensive_data.xlsx'):
        """保存最终结果"""
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False, encoding='utf-8')
        
        # 生成统计报告
        self.generate_report(df)
        
        self.logger.info(f"爬取完成！成功: {self.success_count}, 失败: {len(self.failed_cities)}")
        self.logger.info(f"结果已保存: {filename}")
    
    def generate_report(self, df):
        """生成数据完整性报告"""
        report = {
            'total_cities': len(df),
            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_completeness': {}
        }
        
        # 统计各字段完整性
        fields = ['population', 'area', 'gdp', 'air_quality', 'pm25', 
                 'history_years', 'cultural_sites', 'subway_lines', 
                 'climate_type', 'avg_temperature', 'local_cuisine']
        
        for field in fields:
            if field in df.columns:
                non_empty = df[field].replace('', np.nan).notna().sum()
                completion_rate = (non_empty / len(df)) * 100 if len(df) > 0 else 0
                report['data_completeness'][field] = {
                    'completed': int(non_empty),
                    'completion_rate': f"{completion_rate:.2f}%"
                }
        
        # 保存报告
        with open('city_crawl_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info("数据完整性报告已生成: city_crawl_report.json")
    
    def close(self):
        """关闭浏览器"""
        if hasattr(self, 'driver'):
            self.driver.quit()
    
    def run_crawler(self, city_list=None, start_index=0, output_file='city_data.xlsx'):
        """运行爬虫的主入口"""
        if city_list is None:
            city_list = CITIES_352
            
        try:
            # 爬取数据
            results = self.batch_crawl_cities(city_list, start_index)
            
            if results:
                # 数据验证和清洗
                df = pd.DataFrame(results)
                cleaned_df = self.validator.clean_data(df)
                
                # 保存清洗后的数据
                cleaned_df.to_excel(output_file, index=False, encoding='utf-8')
                
                # 生成数据质量报告
                report = self.validator.generate_data_report(cleaned_df)
                with open('data_quality_report.json', 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                print(f"爬取完成！共处理 {len(results)} 个城市")
                print(f"数据已保存到: {output_file}")
                print(f"数据质量报告: data_quality_report.json")
                
            return results
            
        except KeyboardInterrupt:
            print("用户中断爬取...")
            return []
        except Exception as e:
            print(f"爬取过程中出现错误: {e}")
            return []
        finally:
            self.close()

# 使用示例
if __name__ == "__main__":
    # 创建爬虫实例
    crawler = CityDataCrawler(headless=False)
    
    # 可以选择爬取部分城市进行测试
    test_cities = ['北京', '上海', '广州', '深圳', '杭州']
    
    # 运行爬虫
    results = crawler.run_crawler(
        city_list=test_cities,  # 或者使用 None 爬取全部352个城市
        start_index=0,
        output_file='city_comprehensive_data.xlsx'
    )