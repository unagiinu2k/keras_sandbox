from selenium import webdriver
import chromedriver_binary

import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_a_element(driver, xpath, caster):
    items = driver.find_elements_by_xpath(xpath)

    if len(items) > 0:
        try:
            return (caster(items[0].text))
        except:
            return (None)
    return (None)

options = webdriver.ChromeOptions()
options.add_argument(r"user-data-dir=C:\Users\t\PycharmProjects\keras_sandbox\chrome_profile")
driver = webdriver.Chrome(chrome_options=options)
driver.get('https://order.yodobashi.com/yc/login/index.html')

driver.get('https://order.yodobashi.com/yc/orderhistory/index.html')
#driver.find_elements_by_xpath('select[@id = "selectedPerido')
drop_downs = driver.find_elements_by_id('selectedPeriod')
driver.find_element_by_xpath("//select[@id='selectedPeriod']/option[text()='全てのご注文']").click()
driver.find_elements_by_link_text('検索')[1].click()
orders = list()
order_headers = list()
is_go_next = True
order_frame_xpath = '//div[@class = "orderList"]'
while is_go_next:
    time.sleep(5)

    run_test = WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.XPATH, order_frame_xpath)))
    #run_test.click()
    #order_elements = driver.find_elements_by_xpath(order_frame_xpath)
    time.sleep(5)
    order_elements = driver.find_elements_by_xpath(order_frame_xpath)

    for o in order_elements:
        if False:
            o = order_elements[2]
        run_elements = o.find_elements_by_xpath('.//a/strong/span')
        run_element_header = o.find_elements_by_xpath('./div/div/ul/li/span') #[@class = "hznList"]')
        order_headers.append([x.text for x in run_element_header])
        #hznList
        #fs12
        orders.append([x.text for x in run_elements])

    next_elements = driver.find_elements_by_link_text('次のページ')
    if len(next_elements) > 0:
        to_click = WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.LINK_TEXT, '次のページ')))
        to_click.click()
        #next_elements[0].click()
    else:
        is_go_next = False
import pickle

with open('data/orders.pkl', 'wb') as f:
    pickle.dump(orders, f)

with open('data/order_headers.pkl', 'wb') as f:
    pickle.dump(order_headers, f)
#from selenium.webdriver.common.keys import Keys


order_detail_list = list()
for h in order_headers:
    if False:
        h = order_headers[13]

    driver.get('https://order.yodobashi.com/yc/login/index.html')
    time.sleep(5)
    run_input_fields = driver.find_elements_by_id("orderNo")
    run_input_fields[0].send_keys(h[1])
    driver.find_elements_by_link_text('検索')[0].click()
    time.sleep(5)
    #run_input_fields[0].send_keys(Keys.ENTER)
    item_list_elements = driver.find_elements_by_xpath('//div[@class = "cartItemList"]')#//*[@id="contents"]/div[4]/div/div[2]/div/div
    order_detail_elements = item_list_elements[0].find_elements_by_xpath('.//div[contains(@class , "orderDetailBlock")]')

    for o in order_detail_elements:
        run_product = dict()
        if False:
            o = order_detail_elements[0]
        run_url = o.find_elements_by_xpath('.//div/table/tbody/tr/td/table/tbody/tr/td/p/a')[0].get_attribute('href')
        product_link_text_elements = o.find_elements_by_xpath('.//div/table/tbody/tr/td/table/tbody/tr/td/p/a/strong/span')
        run_product['link_text'] = [x.text for x in product_link_text_elements]
        #//*[@id="contents"]/div[4]/div/div[2]/div/div/div/div/table/tbody/tr[1]/td[2]/table/tbody/tr/td[1]/p/a/strong/span[4]
        run_product['url'] = run_url
        run_unit_price = o.find_elements_by_xpath('.//td[@class = "ecPriceArea"]')[0].text
        run_product['unit_price'] = run_unit_pricerun_unit_price = o.find_elements_by_xpath('.//td[@class = "ecPriceArea"]')[0].text

        run_quantity = o.find_elements_by_xpath('.//td[@class = "ecQuantityArea"]')[0].text
        run_product['quantity'] = run_quantity
        run_product['date'] = h[0]
        run_product['order_no'] = h[1]
        run_product['delivery_status'] = h[2]
        order_detail_list.append(run_product)

with open('data/order_details.pkl', 'wb') as f:
    pickle.dump(order_detail_list, f)

driver.close()

makers = list()
unit_prices = list()
quantities = list()
urls = list()
product_names = list()
for i in order_detail_list:

    if False:
        i = order_detail_list[0]
    i['date']
    run_unit_price = i['unit_price']
    if run_unit_price != "":
        makers.append(i['link_text'][0])
        product_names.append(i['link_text'][3])
        unit_prices.append(int(i['unit_price'][1:].replace(",","")))
        quantities.append(int(i['quantity']))
        urls.append(i['url'])

import pandas as pd
order_detail_df = pd.DataFrame(dict(maker = makers , unit_price = unit_prices , quantity = quantities , url = urls , product_name = product_names))
order_detail_df.to_pickle('data/order_details_df.pkl')
from collections import Counter
Counter(order_detail_df.maker).most_common()

order_detail_df['payment'] = order_detail_df.quantity * order_detail_df.unit_price
order_detail_df.groupby('maker').agg({'payment':'sum'}).sort_values('payment')
makers = list()
product_names = list()
for o in orders:
    if False:
        o = orders[3]
    for j in range(int(len(o) / 4)):
        makers.append(o[j * 4])
        product_names.append(o[j * 4 + 3])
