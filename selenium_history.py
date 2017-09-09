from selenium import webdriver
import chromedriver_binary
import time


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

order_elements = driver.find_elements_by_xpath('//div[@class = "orderList"]')
orders = list()
order_headers = list()
for o in order_elements:
    if False:
        o = order_elements[1]
    run_elements = o.find_elements_by_xpath('.//a/strong/span')
    run_element_header = o.find_elements_by_xpath('./div/div/ul/li/span') #[@class = "hznList"]')
    order_headers.append([x.text for x in run_element_header])
    #hznList
    #fs12
    orders.append([x.text for x in run_elements])


##will delete below
tmp = drop_downs[0]
tmp.s
drop_downs[0].selectByValue('全てのご注文')
#driver.find_elements_by_link_text('カテゴリ')
#driver.find_elements_by_xpath("//div[@class = 'action-bar-dropdown']")
category_dropdowns = driver.find_elements_by_xpath("//button[@aria-controls = 'action-dropdown-children-カテゴリ']")
category_dropdowns[0].click()
L0_elements = category_dropdowns[0].find_elements_by_xpath("//descendant::a[@class = 'child-submenu-link']")
L0_links = [x.get_attribute("href") for x in L0_elements]
print('number of categories = {0}'.format(len(L0_links)))



for i0 , s0 in enumerate(L0_elements):
    if False:
        i0 = 1
        s0 = L0_elements[1]
    s0.click()
    L1_elements = driver.find_elements_by_link_text('もっと見る')
    for i1 , s1 in enumerate(L1_elements):
        if False:
            i1 = 1
            s1 = L1_elements[1]
        s1.click()
        #L2_elements = driver.find_elements_by_link_text('もっと見る')
        is_continue = True
        while is_continue:
            show_mores = driver.find_elements_by_xpath("//button[@id ='show-more-button']")
            if len(show_mores) == 0:
                is_continue = False
            else:
                time.sleep(5)
                show_mores[0].click()
            if False:
                scores = driver.find_elements_by_xpath("//div[@class = 'score']")
                score = None
                if len(scores) > 0:
                    try:
                        score = float(scores[0].text)
                    except:
                        pass

                review_ns = driver.find_elements_by_xpath("//span[@class = 'reviews-num']")
                review_n = None
                if len(review_ns) > 0:
                    try:
                        review_n = int(review_ns[0].text.replace(',',''))
                    except:
                        pass
                date_updates = driver.find_elements_by_xpath("//div[@itemprop = 'datePublished']")
                date_update = None
                if len(date_updates) > 0:
                    try:
                        date_updte = date_updates[0].text
                    except:
                        pass
                def get_a_element(driver , xpath , caster):
                    items = driver.find_elements_by_xpath(xpath)
                    if len(items) > 0:
                        try:
                            return(caster(items[0].text))
                        except:
                            return(None)
                    return(None)


            # get rating, maker name, maker url, review count etc.. for each product
            score = get_a_element(driver, "//div[@class = 'score']" , lambda x:float(x))
            reviews_num = get_a_element(driver , "//span[@class = 'reviews-num']" , lambda x:int(x.replace(',' , '')))
            datePublished = get_a_element(driver, "//div[@itemprop = 'datePublished']" ,lambda x:x)
            numDownloads = get_a_element(driver, "//div[@itemprop = 'numDownloads']", lambda x: x)
            contentRating = get_a_element(driver, "//div[@itemprop = 'contentRating']", lambda x: x)
            driver.find_elements_by_link_text('提供元')
            meta_infos = driver.find_elements_by_xpath("//div[contains(@class , 'meta-info')]")

            meta_info_list = [x.text for x in meta_infos]






driver.close()