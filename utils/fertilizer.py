fertilizer_dic = {
    'NHigh': """The N value of soil is high and might give rise to weeds.
    <br/> Please consider the following suggestions:

    <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

    <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

    <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

    <br/>4. Plant ‘green manure’ crops like cabbage, corn and brocolli

    <br/>5. <i>Use mulch (wet grass) while growing crops</i> - Mulch can also include sawdust and scrap soft woods""",

    'Nlow': """The N value of your soil is low.
    <br/> Please consider the following suggestions:
    <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

    <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

    <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

    <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

    <br/>5. Add composted manure to the soil.

    <br/>6. Plant Nitrogen fixing plants like peas or beans.

    <br/>7. <i>Use NPK fertilizers with high N value.</i>

    <br/>8. <i>Do nothing</i> – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

    'PHigh': """The P value of your soil is high.
    <br/> Please consider the following suggestions:

    <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

    <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

    <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

    <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

    <br/>5. Use crop rotations to decrease high phosphorous levels""",

    'Plow': """The P value of your soil is low.
    <br/> Please consider the following suggestions:

    <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

    <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

    <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

    <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

    <br/>5. <i>Manure</i> – as with compost, manure can be an excellent source of phosphorous for your plants.

    <br/>6. <i>Clay soil</i> – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

    <br/>7. <i>Ensure proper soil pH</i> – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

    <br/>8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.

    <br/>9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

    'KHigh': """The K value of your soil is high.
    <br/> Please consider the following suggestions:

    <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

    <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

    <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

    <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

    <br/>5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

    <br/>6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.""",

    'Klow': """The K value of your soil is low.
    <br/>Please consider the following suggestions:

    <br/><br/>1. Mix in muricate of potash or sulphate of potash
    <br/>2. Try kelp meal or seaweed
    <br/>3. Try Sul-Po-Mag
    <br/>4. Bury banana peels an inch below the soil surface
    <br/>5. Use Potash fertilizers since they contain high values potassium
    """
}


fertilizer_dic_mr = {
    'NHigh': """मातीतील नायट्रोजन (N) चे प्रमाण जास्त आहे आणि त्यामुळे तण वाढण्याची शक्यता आहे.
    <br/> कृपया खालील सूचना विचारात घ्या:

    <br/><br/>1. <i>शेणखत / सेंद्रिय खत</i> – शेणखत वापरणे हा मातीमध्ये नायट्रोजन संतुलित करण्याचा सोपा मार्ग आहे. मात्र विविध खतांमध्ये नायट्रोजनचे प्रमाण वेगवेगळे असते, त्यामुळे काळजीपूर्वक वापरा.

    <br/>2. <i>कॉफीचे चोथरे</i> – कॉफी ग्राइंड्स हे नायट्रोजनयुक्त हिरवे कंपोस्ट मानले जातात. ते विघटित झाल्यावर मातीला नायट्रोजन मिळते. यामुळे पाण्याचा निचरा सुधारणेसही मदत होते.

    <br/>3. <i>नायट्रोजन स्थिर करणारी पिके लावा</i> – वाटाणा, शेंगा, सोयाबीन यांसारखी कडधान्य पिके मातीतील नायट्रोजन वाढवण्यास मदत करतात.

    <br/>4. कोबी, मका, ब्रोकोली यांसारखी <i>ग्रीन मॅन्युअर</i> पिके लावा.

    <br/>5. <i>मल्च (ओले गवत)</i> वापरा – मल्चमध्ये भूसा किंवा लाकडाचे बारीक तुकडेही वापरता येतात.""",

    'Nlow': """तुमच्या मातीतील नायट्रोजन (N) चे प्रमाण कमी आहे.
    <br/> कृपया खालील सूचना विचारात घ्या:

    <br/><br/>1. <i>भूसा किंवा बारीक लाकडी भुगा मातीमध्ये मिसळा</i> – यातील कार्बन नायट्रोजन शोषून घेण्यास मदत करतो आणि मातीचे संतुलन सुधारतो.

    <br/>2. <i>जास्त नायट्रोजन घेणारी पिके लावा</i> – टोमॅटो, मका, ब्रोकोली, कोबी आणि पालक ही नायट्रोजन जास्त वापरणारी पिके आहेत.

    <br/>3. <i>पाणी द्या</i> – मातीला भरपूर पाणी दिल्यास नायट्रोजन मातीच्या खालच्या थरात जाऊ शकतो.

    <br/>4. <i>साखर मर्यादित प्रमाणात वापरा</i> – काही अभ्यासानुसार साखर वापरल्यास मातीतील नायट्रोजनचे प्रमाण कमी होण्यास मदत होऊ शकते.

    <br/>5. कुजलेले शेणखत मातीमध्ये मिसळा.

    <br/>6. वाटाणा किंवा शेंगांप्रमाणे नायट्रोजन स्थिर करणारी पिके लावा.

    <br/>7. <i>NPK खतांमध्ये N चे प्रमाण जास्त असलेले खत वापरा.</i>

    <br/>8. <i>कधी कधी काहीही करू नका</i> – जर तुमची पिके आधीच चांगली पाने देत असतील, तर तीच मातीतील नायट्रोजन वापरून संतुलन साधू शकतात.""",

    'PHigh': """तुमच्या मातीतील फॉस्फरस (P) चे प्रमाण जास्त आहे.
    <br/> कृपया खालील सूचना विचारात घ्या:

    <br/><br/>1. <i>शेणखत टाळा</i> – शेणखतात फॉस्फरसचे प्रमाण जास्त असू शकते, त्यामुळे त्याचा वापर कमी करा.

    <br/>2. <i>फॉस्फरस नसलेले खत वापरा</i> – 10-0-10 सारखे खत वापरल्यास इतर अन्नद्रव्ये मिळतील पण फॉस्फरस वाढणार नाही.

    <br/>3. <i>मातीला भरपूर पाणी द्या</i> – त्यामुळे फॉस्फरस मातीबाहेर जाण्यास मदत होऊ शकते. हा शेवटचा उपाय म्हणून वापरा.

    <br/>4. शेंगा, वाटाणा यांसारखी नायट्रोजन स्थिर करणारी पिके लावा ज्यामुळे फॉस्फरस न वाढवता नायट्रोजन वाढेल.

    <br/>5. पीक फेरपालट (crop rotation) वापरा जेणेकरून फॉस्फरसचे जास्त प्रमाण कमी होईल.""",

    'Plow': """तुमच्या मातीतील फॉस्फरस (P) चे प्रमाण कमी आहे.
    <br/> कृपया खालील सूचना विचारात घ्या:

    <br/><br/>1. <i>बोन मील</i> – हे प्राण्यांच्या हाडांपासून बनवलेले असून फॉस्फरसचा जलद स्रोत आहे.

    <br/>2. <i>रॉक फॉस्फेट</i> – हा हळूहळू कार्य करणारा स्रोत आहे. माती त्याचे फॉस्फरस मध्ये रूपांतर करते.

    <br/>3. <i>फॉस्फरसयुक्त खते</i> – 10-20-10 सारख्या NPK खतांचा वापर करा ज्यात फॉस्फरस जास्त असतो.

    <br/>4. <i>सेंद्रिय कंपोस्ट</i> – चांगल्या दर्जाचे कंपोस्ट मातीतील फॉस्फरस वाढवते.

    <br/>5. <i>शेणखत</i> – हे देखील फॉस्फरसचा चांगला स्रोत आहे.

    <br/>6. <i>चिकणमाती</i> – मातीमध्ये चिकणमातीचे कण मिसळल्याने फॉस्फरस टिकून राहण्यास मदत होते.

    <br/>7. <i>योग्य pH राखा</i> – 6.0 ते 7.0 या pH मध्ये वनस्पती फॉस्फरस उत्तमरीत्या शोषतात.

    <br/>8. जर pH कमी असेल तर चुना किंवा पोटॅशियम कार्बोनेट वापरा.

    <br/>9. जर pH जास्त असेल तर सेंद्रिय पदार्थ वाढवा किंवा अमोनियम सल्फेटसारखी आम्लीय खते वापरा.""",

    'KHigh': """तुमच्या मातीतील पोटॅशियम (K) चे प्रमाण जास्त आहे.
    <br/> कृपया खालील सूचना विचारात घ्या:

    <br/><br/>1. <i>माती खोलवर सैल करा</i> आणि भरपूर पाणी द्या, जेणेकरून पाण्यात विरघळणारे पोटॅशियम खाली जाईल. माती वाळल्यानंतर ही प्रक्रिया पुन्हा करा.

    <br/>2. <i>माती चाळून घ्या</i> आणि शक्य तितके दगड वेगळे करा. काही खनिजांमधून हळूहळू पोटॅशियम सुटत राहते.

    <br/>3. पोटॅशियम जास्त असलेले रासायनिक खत देणे थांबवा. शेवटचा आकडा 0 असलेले खत वापरा.

    <br/>4. अंड्याची टरफले, शिंपल्यांची पूड, लाकडाची राख किंवा सॉफ्ट रॉक फॉस्फेट मातीमध्ये मिसळा. थोडे सेंद्रिय कंपोस्टही मिसळा.

    <br/>5. K कमी असलेली NPK खते किंवा सेंद्रिय खते वापरा.

    <br/>6. कडधान्य वर्गातील आच्छादन पिके (cover crops) लावा. त्यामुळे नायट्रोजन मिळेल पण फॉस्फरस किंवा पोटॅशियम वाढणार नाही.""",

    'Klow': """तुमच्या मातीतील पोटॅशियम (K) चे प्रमाण कमी आहे.
    <br/> कृपया खालील सूचना विचारात घ्या:

    <br/><br/>1. म्युरेट ऑफ पोटॅश किंवा सल्फेट ऑफ पोटॅश मातीमध्ये मिसळा.
    <br/>2. केल्प मील किंवा समुद्री शैवाल वापरून पाहा.
    <br/>3. सल्-पो-मॅग वापरून पाहा.
    <br/>4. केळीची साले मातीच्या पृष्ठभागाखाली साधारण एक इंच खोल पुरा.
    <br/>5. पोटॅश खते वापरा कारण त्यामध्ये पोटॅशियमचे प्रमाण जास्त असते.
    """
}