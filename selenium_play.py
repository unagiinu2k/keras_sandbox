from selenium import webdriver
import chromedriver_binary
import time
driver = webdriver.Chrome()
driver.get('https://play.google.com/store/apps')
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

            #get rating, maker name, maker url, review count etc.. for each product




driver.close()