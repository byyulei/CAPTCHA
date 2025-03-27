import requests
import re, execjs
from urllib import parse
import json
import ddddocr
import random
class Generate_trajectory:
    @staticmethod
    def __ease_out_expo(sep):
        if sep == 1:
            return 1
        else:
            return 1 - pow(2, -10 * sep)

    @staticmethod
    def __generate_y():
        init_y = random.randint(10, 30)
        y = [init_y]
        for i in range(20):
            _ = random.randint(-1, 1)
            y += [y[-1] + _] * random.randint(5, 10)
        return y

    @classmethod
    def get_slide_track(cls, distance):
        """
        根据滑动距离生成滑动轨迹
        :param distance: 需要滑动的距离
        """

        if not isinstance(distance, int) or distance < 0:
            raise ValueError(f"distance类型必须是大于等于0的整数: distance: {distance}, type: {type(distance)}")
        count_x = random.randint(10, 23)
        # 共记录count次滑块位置信息
        count = count_x + int(distance / 20)
        # 初始化滑动时间
        t = random.randint(50, 100)
        # 记录上一次滑动的距离
        _x = 0
        _y = cls.__generate_y()

        # 初始化轨迹列表
        slide_track = [
            '|'.join([str(random.randint(10, 30)), str(_y.pop(0)), str(0)])
        ]

        for i in range(count):
            # 已滑动的横向距离
            x = round(cls.__ease_out_expo(i / count) * distance)
            # 滑动过程消耗的时间
            t += random.randint(25, 40)
            if x == _x:
                continue
            slide_track.append('|'.join([str(x), str(_y[i]), str(t)]))
            _x = x
        return ','.join(slide_track)
class kuaishou_captchaToken():
    def __init__(self):
        self.headers = {
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
            "accept": "*/*",
            "content-type": "application/json",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36",
            "Origin": "https://www.kuaishou.com",
            "Referer": "https://www.kuaishou.com/profile/3xgefifvenffbku",
            "Accept-Language": "zh-CN,zh;q=0.9"
        }
        self.session = requests.session()
        self.session.get('https://www.kuaishou.com/profile/3xgefifvenffbku', headers=self.headers)
        self.captchaSession = None
        self.captchaSn = None
        self.disY = None
        self.distance = None
        self.trajectory = None
        self.data_cr=None
        self._get_captchaSession()#拿回captchaSession参数
        self._load_encrypt()#加载js
        self._get_image()

    def _load_encrypt(self):
        """
            加载js
            :return:
            """
        with open("captchaToken_encrypt.js", "rb") as f:
            js = f.read().decode()
        self.data_cr = execjs.compile(js)


    # 拿回captchaSession参数
    def _get_captchaSession(self):
        url = "https://www.kuaishou.com/graphql"
        data = {
            "operationName": "visionProfilePhotoList",
            "variables": {
                "userId": "3xgefifvenffbku",
                "pcursor": "",
                "page": "profile"
            },
            "query": "fragment photoContent on PhotoEntity {\n  id\n  duration\n  caption\n  likeCount\n  viewCount\n  realLikeCount\n  coverUrl\n  photoUrl\n  photoH265Url\n  manifest\n  manifestH265\n  videoResource\n  coverUrls {\n    url\n    __typename\n  }\n  timestamp\n  expTag\n  animatedCoverUrl\n  distance\n  videoRatio\n  liked\n  stereoType\n  profileUserTopPhoto\n  __typename\n}\n\nfragment feedContent on Feed {\n  type\n  author {\n    id\n    name\n    headerUrl\n    following\n    headerUrls {\n      url\n      __typename\n    }\n    __typename\n  }\n  photo {\n    ...photoContent\n    __typename\n  }\n  canAddComment\n  llsid\n  status\n  currentPcursor\n  __typename\n}\n\nquery visionProfilePhotoList($pcursor: String, $userId: String, $page: String, $webPageArea: String) {\n  visionProfilePhotoList(pcursor: $pcursor, userId: $userId, page: $page, webPageArea: $webPageArea) {\n    result\n    llsid\n    webPageArea\n    feeds {\n      ...feedContent\n      __typename\n    }\n    hostName\n    pcursor\n    __typename\n  }\n}\n"
        }
        response = self.session.post(url, headers=self.headers, data=json.dumps(data))
        self.captchaSession = re.findall('captchaSession=(.*)&type', response.text)[0]

    # 拿到图片链接
    def _get_image(self):

        url = "https://captcha.zt.kuaishou.com/rest/zt/captcha/sliding/config"
        data = {
            "captchaSession": self.captchaSession
        }
        response = self.session.post(url,data=data)
        self.captchaSn = json.loads(response.text)['captchaSn']
        ditu = json.loads(response.text)['bgPicUrl']
        quekou = json.loads(response.text)['cutPicUrl']
        self.disY = round(int(json.loads(response.text)['disY']) * 0.469)
        params = {
            "captchaSn": self.captchaSn
        }
        name1 = self.session.get(ditu, params=params).content
        name2 = self.session.get(quekou, params=params).content
        distance = self.slide_match(name1, name2)
        distance_ = int(round(int(distance['target'][0]) * 0.469, 0) - 5)
        self.distance = distance_
        # 轨迹
        self.trajectory = Generate_trajectory().get_slide_track(int(distance_ * 3.42))

    # 用ddddocr识别滑块距离
    def slide_match(self, target_bytes, background_bytes):
        det = ddddocr.DdddOcr(det=False, ocr=False, show_ad=False)
        res = det.slide_match(target_bytes, background_bytes, simple_target=True)
        return res

    # 请求验证滑块
    def get_verify(self):
        captcha_data = {
            "captchaSn": self.captchaSn,
            "bgDisWidth": 316,
            "bgDisHeight": 184,
            "cutDisWidth": 56,
            "cutDisHeight": 56,
            "relativeX": self.distance,
            "relativeY": self.disY,
            "trajectory": self.trajectory,
            # "滑块拖动轨迹",
            "gpuInfo": "{\"glRenderer\":\"WebKit WebGL\",\"glVendor\":\"WebKit\",\"unmaskRenderer\":\"ANGLE (Intel, Intel(R) HD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11-31.0.101.2111)\",\"unmaskVendor\":\"Google Inc. (Intel)\"}",
            # '显卡信息，可以写死',
            "captchaExtraParam": "{\"ua\":\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36\",\"userAgent\":\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36\",\"timeZone\":\"UTC+8\",\"language\":\"zh-CN\",\"cpuCoreCnt\":\"4\",\"platform\":\"Win32\",\"riskBrowser\":\"false\",\"webDriver\":\"false\",\"exactRiskBrowser\":\"false\",\"webDriverDeep\":\"false\",\"exactRiskBrowser2\":\"false\",\"webDriverDeep2\":\"false\",\"battery\":\"1\",\"plugins\":\"170b76c0d6cbaed3cb42746b06cae5eae\",\"resolution\":\"1920x1080\",\"pixelDepth\":\"24\",\"colorDepth\":\"24\",\"canvasGraphFingerPrint\":\"184d2a5cb7dd0dbef53dd603e607d58f8\",\"canvasGraph\":\"184d2a5cb7dd0dbef53dd603e607d58f8\",\"canvasTextFingerPrintEn\":\"10988b111dee10a3ace1f10536e3a0eee\",\"canvasTextEn\":\"10988b111dee10a3ace1f10536e3a0eee\",\"canvasTextFingerPrintZh\":\"13ffb1298cbb06fbd4cbf7a29b627d240\",\"canvasTextZh\":\"13ffb1298cbb06fbd4cbf7a29b627d240\",\"webglGraphFingerPrint\":\"14dc51ccd006f78c818999c374ac62402\",\"webglGraph\":\"14dc51ccd006f78c818999c374ac62402\",\"webglGPUFingerPrint\":\"1108f3efe4bed369a31b6475af6c38f30\",\"webglGpu\":\"1108f3efe4bed369a31b6475af6c38f30\",\"cssFontFingerPrintEn\":\"10a344f5534d5b367655c7f90f04de717\",\"fontListEn\":\"10a344f5534d5b367655c7f90f04de717\",\"cssFontFingerPrintZh\":\"16c1334aeae228bca19e18632c8472a52\",\"fontListZh\":\"16c1334aeae228bca19e18632c8472a52\",\"voiceFingerPrint\":\"1dd96cac4e826abdbbe261dc4f3a08292\",\"audioTriangle\":\"1dd96cac4e826abdbbe261dc4f3a08292\",\"nativeFunc\":\"1973dcbb27a04c3a2ee240d9d2549e105\",\"key1\":\"web_11896f467df30247503494240be3a7a2\",\"key2\":1682662120147,\"key3\":\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36\",\"key4\":\"20030107\",\"key5\":\"zh-CN\",\"key6\":\"Gecko\",\"key7\":1920,\"key8\":1080,\"key9\":1920,\"key10\":1040,\"key11\":360,\"key12\":360,\"key13\":878,\"key14\":1358,\"key15\":\"00000111\",\"key16\":1,\"key17\":1,\"key18\":[],\"key19\":{},\"key20\":[],\"key21\":{},\"key22\":[],\"key23\":{},\"key24\":[],\"key25\":{},\"key26\":{\"key27\":[\"0,1,3076,267,321,prepare1\",\"1,1,3087,256,314,prepare1\",\"2,1,3092,242,310,prepare1\",\"3,1,3102,230,304,prepare1\",\"4,1,3108,219,300,prepare1\",\"5,1,3116,209,294,prepare1\",\"6,1,3124,199,291,prepare1\",\"7,1,3135,191,288,prepare1\",\"8,1,3158,173,281,prepare1\",\"9,1,3164,169,278,prepare1\",\"10,3,3919,44,232\",\"11,1,3940,44,233,prepare2\",\"12,1,3956,45,234,prepare2\",\"13,1,3964,46,235,prepare2\",\"14,1,3972,47,235,prepare2\",\"15,1,3980,51,237,prepare2\",\"16,1,3988,54,237,prepare2\",\"17,1,3996,58,239,prepare2\",\"18,1,4004,61,239,prepare2\",\"19,1,4012,65,239,prepare2\",\"20,1,4020,68,239,prepare2\",\"21,4,4484,193,234\",\"22,2,4688,193,234,prepare3\",\"23,1,4689,262,321,prepare3\"],\"key28\":[],\"key29\":[],\"key30\":[],\"key31\":{\"prepare1\":\"9,1,3164,169,278\",\"prepare2\":\"20,1,4020,68,239\",\"prepare3\":\"23,1,4689,262,321\"},\"key32\":{},\"key33\":{},\"key34\":{}},\"key35\":\"7ebc4735321e3b0c225c1e489d2adb1b\",\"key36\":\"f22a94013fc94e90e2af2798023a1985\",\"key37\":1,\"key38\":\"not support\",\"key39\":4}"
            # '浏览器信息和指纹什么的'
        }
        captcha_str = ''
        for k, v in captcha_data.items():
            captcha_str += f'&{k}={parse.quote(str(v))}'
        captcha_txt = captcha_str[1:].replace('/', '%2F')
        data_crack = self.data_cr.call("get_data",captcha_txt)
        json_data = {'verifyParam': data_crack}
        response = self.session.post('https://captcha.zt.kuaishou.com/rest/zt/captcha/sliding/kSecretApiVerify',json=json_data).json()
        return response

while True:
    print(kuaishou_captchaToken().get_verify())




