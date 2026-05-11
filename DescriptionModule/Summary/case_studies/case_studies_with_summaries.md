# Case studies — top-5 with video summaries
Source files:
- `RetrievalModule/outputs/topk_baseline.json` (zero-shot embedder)
- `RetrievalModule/outputs/topk_test_phase2_ck900.json` (P2 ck-900 fine-tuned embedder, also = stage1 input for rerank)
- `RetrievalModule/outputs/rerank_v6_ck50_multi.json` (reranker v6 ck-50 multimodal)
- Summaries: `DescriptionModule/Summary/test_summaries_v2.jsonl` (`full_summary` field)


---

## S1 — SUCCESS — full pipeline win (ZS miss → FT@4 → RR@1)

**Query:** *In a residential area, a car was parked on the side of the road, and a man in blue stuck the card in his hand to the windshield of the car.*

**Ground truth:** `Testing_Normal_Videos_Anomaly/Normal_Videos_656`

**GT video summary** (`anomaly_type`: *None*):
> The video begins with a white car moving across the screen. Subsequently, a man in blue clothes and a woman in blue clothes walked across the screen. Later, a woman in a white dress and a man in blue clothes walked across the screen. The video does not show any unusual or violent behavior.

### Stage 0 — ZS embedder top-5 (GT rank: miss)

  **1. `Stealing/Stealing019`**  *(anomaly_type: suspicious activity)*
   > A man in a black shirt and hat approached a white car and opened the trunk. Later, a man in a blue shirt and hat opened the same white car's trunk. Both men retrieved objects from the trunk, with one man subsequently discarding a black object into the bushes. The individuals were near a white car.

  **2. `RoadAccidents/RoadAccidents128`**  *(anomaly_type: unusual vehicle behavior)*
   > A white car repeatedly entered and exited the road. Simultaneously, a man in black ran across the road. The events occurred in an area with vehicles and pedestrians. This presented a potentially hazardous situation.

  **3. `RoadAccidents/RoadAccidents001`**  *(anomaly_type: physical altercation)*
   > A man in blue clothes and a woman in black clothes entered a house. Subsequently, the man in blue clothes was observed lying on the ground. Two men in black clothes were observed engaging in a physical altercation. The clips depict movement and interaction between individuals.

  **4. `Shooting/Shooting043`**  *(anomaly_type: violent encounter)*
   > A man in a red jacket and blue jeans approached a car. Another man in blue clothes entered the car. Subsequently, a man in black clothes took out a gun and pointed it at the man in blue clothes. A man in a red shirt and blue jeans approached the car and grabbed the car's front bumper.

  **5. `Burglary/Burglary076`**  *(anomaly_type: suspicious activity)*
   > A man in black clothes walked towards a car. Another man in black clothes and a hat walked towards the car. A man in blue jeans and a black jacket ran from the bottom to the top of the screen. A man in black clothes and a black hat walked towards a house. These actions occurred near the vehicle.

### Stage 1 — FT embedder (P2 ck-900) top-5 (GT rank: 4)

  **1. `Stealing/Stealing019`**  *(anomaly_type: suspicious activity)*
   > A man in a black shirt and hat approached a white car and opened the trunk. Later, a man in a blue shirt and hat opened the same white car's trunk. Both men retrieved objects from the trunk, with one man subsequently discarding a black object into the bushes. The individuals were near a white car.

  **2. `Testing_Normal_Videos_Anomaly/Normal_Videos_051`**  *(anomaly_type: none)*
   > The video depicts a white SUV driving down the street. A person wearing a white shirt and black pants walks to the right side of the road. A black car and a white car are also visible driving on the road. The scene is uneventful and shows no unusual activities.

  **3. `Vandalism/Vandalism028`**  *(anomaly_type: unusual behavior)*
   > A man wearing white clothes and a white hat entered a room and opened the door. Subsequently, the man walked out of the house and moved toward a vehicle. He repeatedly exited the door. The man was consistently observed walking and moving away from the camera.

★ **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_656`**  *(anomaly_type: None)*
   > The video begins with a white car moving across the screen. Subsequently, a man in blue clothes and a woman in blue clothes walked across the screen. Later, a woman in a white dress and a man in blue clothes walked across the screen. The video does not show any unusual or violent behavior.

  **5. `Testing_Normal_Videos_Anomaly/Normal_Videos_896`**  *(anomaly_type: normal)*
   > The video shows a normal traffic scene. A silver car drives past a red car. A woman in black clothes and a child in a red jacket are walking on the road. A man in black clothes is walking on the sidewalk, and another man in black clothes is pushing a stroller.

### Stage 2 — Reranker v6 ck-50 top-5 (GT rank: 1)

★ **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_656`**  *(anomaly_type: None; rr_score=0.0427, stage1_score=0.3874)*
   > The video begins with a white car moving across the screen. Subsequently, a man in blue clothes and a woman in blue clothes walked across the screen. Later, a woman in a white dress and a man in blue clothes walked across the screen. The video does not show any unusual or violent behavior.

  **2. `Robbery/Robbery050`**  *(anomaly_type: Potential threat; rr_score=0.0240, stage1_score=0.3489)*
   > A white SUV is parked on a brick driveway. Two men approach the vehicle; one opens the trunk while the other stands nearby. The men then engage in conversation. One man opens the trunk of the SUV, and the other man observes.

  **3. `Shooting/Shooting037`**  *(anomaly_type: collision; rr_score=0.0223, stage1_score=0.3332)*
   > A black car stopped and then reversed. Subsequently, a black car drove away from the location. A man in a black shirt and black pants approached the black car and pushed it. A black car drove past a green car on the road.

  **4. `Shooting/Shooting004`**  *(anomaly_type: unusual movement; rr_score=0.0183, stage1_score=0.3119)*
   > A man in a black shirt and white pants ran towards a green car, and a woman in purple clothes ran towards the same car. Another man in a black shirt and blue jeans ran towards the camera, and then ran away from the camera. A man in a purple shirt and black pants ran across the road. These individuals moved towards the green car.

  **5. `Stealing/Stealing019`**  *(anomaly_type: suspicious activity; rr_score=0.0164, stage1_score=0.4652)*
   > A man in a black shirt and hat approached a white car and opened the trunk. Later, a man in a blue shirt and hat opened the same white car's trunk. Both men retrieved objects from the trunk, with one man subsequently discarding a black object into the bushes. The individuals were near a white car.


---

## S2 — SUCCESS — Phase 2 fix (ZS@24 → FT@1 → RR@1)

**Query:** *A man moved a large item of merchandise out of the store.*

**Ground truth:** `Shoplifting/Shoplifting039`

**GT video summary** (`anomaly_type`: *unusual behavior*):
> A man wearing a striped shirt and dark pants entered a room, picked up a box, and then exited the room. Another man wearing a plaid shirt and jeans entered a store, walked to the counter, and then left the store. A man wearing a red shirt and black shorts entered a room, carried a box, and then exited the room. A man wearing orange shorts and black shorts entered a room, looked around, and then exited the room.

### Stage 0 — ZS embedder top-5 (GT rank: 24)

  **1. `Shoplifting/Shoplifting033`**  *(anomaly_type: normal)*
   > The video shows a man in a blue shirt and khaki shorts entering a store and approaching a counter. He engages in conversation with a woman wearing a black top and a white hat, who is standing at the counter. The man then walks away from the counter, and the woman continues to talk to him. Subsequently, the man exits the store, and the woman remains at the counter.

  **2. `Testing_Normal_Videos_Anomaly/Normal_Videos_934`**  *(anomaly_type: potential robbery or theft)*
   > The video depicts a man wearing blue clothes walking from the bottom to the top of the screen and then returning to the bottom. Multiple clips show a man in black clothes walking from the bottom to the top of the screen and then returning to the bottom. One clip shows a man in blue clothes and a blue cap entering and exiting a store.

  **3. `Vandalism/Vandalism036`**  *(anomaly_type: unusual behavior)*
   > A man wearing a white shirt and black pants entered the store and interacted with the cashier. Subsequently, he left the store. Other individuals, including a man in black clothes and a woman in white clothes, also entered and exited the store. Employees reacted to the man's actions. The man in a striped shirt and hat also entered, picked up a bag, and left the store.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_100`**  *(anomaly_type: normal)*
   > The video depicts a normal and mundane scene in a store. Several people, including men and women, are observed walking to the counter. Individuals are wearing various colored shirts and pants, such as blue, black, pink, and green. The scene shows people entering and moving within the store environment.

  **5. `Shoplifting/Shoplifting031`**  *(anomaly_type: None)*
   > A man in a white shirt and black pants entered a store and approached the counter. He then left the store. Two men, one in a blue shirt and one in a white shirt, were sitting on the floor. Another man in a white shirt was standing at the counter. The scene appeared to be uneventful.

### Stage 1 — FT embedder (P2 ck-900) top-5 (GT rank: 1)

★ **1. `Shoplifting/Shoplifting039`**  *(anomaly_type: unusual behavior)*
   > A man wearing a striped shirt and dark pants entered a room, picked up a box, and then exited the room. Another man wearing a plaid shirt and jeans entered a store, walked to the counter, and then left the store. A man wearing a red shirt and black shorts entered a room, carried a box, and then exited the room. A man wearing orange shorts and black shorts entered a room, looked around, and then exited the room.

  **2. `Shoplifting/Shoplifting016`**  *(anomaly_type: none)*
   > The video shows a woman wearing a pink top and a purple skirt entering a room. She picks up a box and then exits the room. A woman in a pink top and a black skirt also enters the room and picks up a box before leaving. Additionally, a woman in a red dress walks from the left side of the screen to the right side of the screen and back again.

  **3. `Shoplifting/Shoplifting015`**  *(anomaly_type: none)*
   > The video shows a man wearing white clothes entering a store. He then takes out a box from the store. Finally, the man leaves the store. No unusual events are observed during the sequence.

  **4. `Shoplifting/Shoplifting027`**  *(anomaly_type: discrepancy)*
   > A man wearing a black shirt and white shorts entered a store. He picked up a bottle and then exited the store. The video shows no unusual or suspicious events occurring. The clips depict a man walking into and out of the store.

  **5. `Shooting/Shooting008`**  *(anomaly_type: physical altercation)*
   > A man in black clothes and a cap walked towards the camera, followed by a man in blue clothes and a cap. The man in black clothes picked up a black suitcase and walked away from the camera. A man in blue clothes and a cap moved a black suitcase from the lower right to the lower left corner of the video. A person in a white shirt and black pants exited a store and then left the store.

### Stage 2 — Reranker v6 ck-50 top-5 (GT rank: 1)

★ **1. `Shoplifting/Shoplifting039`**  *(anomaly_type: unusual behavior; rr_score=0.1240, stage1_score=0.4833)*
   > A man wearing a striped shirt and dark pants entered a room, picked up a box, and then exited the room. Another man wearing a plaid shirt and jeans entered a store, walked to the counter, and then left the store. A man wearing a red shirt and black shorts entered a room, carried a box, and then exited the room. A man wearing orange shorts and black shorts entered a room, looked around, and then exited the room.

  **2. `Vandalism/Vandalism015`**  *(anomaly_type: suspicious behavior; rr_score=0.0781, stage1_score=0.3481)*
   > A man wearing gray entered a store and subsequently exited. Another man in a dark suit and white shirt was observed entering and leaving the store. A man in black appeared and interacted with a man in purple. A man in a dark jacket and pants also entered and exited the store.

  **3. `Shoplifting/Shoplifting029`**  *(anomaly_type: normal; rr_score=0.0718, stage1_score=0.3986)*
   > The video shows a man in a black coat entering a store. He looks around the interior. Subsequently, the man exits the store. The scene contains no unusual or out-of-the-ordinary events.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_940`**  *(anomaly_type: none; rr_score=0.0664, stage1_score=0.3596)*
   > The video shows a man wearing white clothes walking into a store and then exiting the store. He is observed walking past a counter with a laptop. The video depicts a mundane scene with no unusual events. The man is seen walking from the left to the right side of the screen and back again.

  **5. `Vandalism/Vandalism036`**  *(anomaly_type: unusual behavior; rr_score=0.0645, stage1_score=0.4002)*
   > A man wearing a white shirt and black pants entered the store and interacted with the cashier. Subsequently, he left the store. Other individuals, including a man in black clothes and a woman in white clothes, also entered and exited the store. Employees reacted to the man's actions. The man in a striped shirt and hat also entered, picked up a bag, and left the store.


---

## S3 — SUCCESS — reranker rescues FT regression (ZS@4 → FT@12 → RR@1)

**Query:** *In a decoration shop, a group of people are looking at decorations.*

**Ground truth:** `Testing_Normal_Videos_Anomaly/Normal_Videos_884`

**GT video summary** (`anomaly_type`: *none*):
> The video depicts a woman in a blue shirt entering a store and walking to the counter. She then exits the store. The scene shows a mundane activity with no unusual events. Multiple clips corroborate this description, showing a similar sequence of events.

### Stage 0 — ZS embedder top-5 (GT rank: 4)

  **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_889`**  *(anomaly_type: none)*
   > The video depicts a normal and uneventful scene. A man in black walks to the counter, and a woman in black walks to the counter. Subsequently, a man in black walks out of the store, and a woman in black walks out of the store. The scene contains no unusual or suspicious events.

  **2. `Testing_Normal_Videos_Anomaly/Normal_Videos_899`**  *(anomaly_type: none)*
   > The video shows a man in black clothes and a woman in black clothes walking out of a store. The woman is wearing a white shirt and a black skirt. Both individuals move from the bottom to the top of the screen. The scene is normal and uneventful, with no unusual events observed.

  **3. `Shoplifting/Shoplifting005`**  *(anomaly_type: physical altercation)*
   > The video begins with a man in a brown jacket entering and exiting a store. Subsequently, a man in a green jacket and black pants engaged in a physical altercation with another man. A man in a brown jacket approached a woman in a black coat and black pants, and then pushed her to the ground.

★ **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_884`**  *(anomaly_type: none)*
   > The video depicts a woman in a blue shirt entering a store and walking to the counter. She then exits the store. The scene shows a mundane activity with no unusual events. Multiple clips corroborate this description, showing a similar sequence of events.

  **5. `Testing_Normal_Videos_Anomaly/Normal_Videos_938`**  *(anomaly_type: none)*
   > The video shows a man in black and a woman in white entering a store. Subsequently, a man in white clothes leaves the store. Later, a woman in white shorts and a man in a white shirt enter the store, and a woman in a white shirt and blue shorts exits the store. Throughout the video, no unusual or suspicious events are observed.

### Stage 1 — FT embedder (P2 ck-900) top-5 (GT rank: 12)

  **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_890`**  *(anomaly_type: normal)*
   > The video shows a busy indoor event, possibly a trade show or exhibition. People are browsing, conversing, and walking around the space. Several individuals are engaged in conversations with others. A booth with a blue tablecloth displays bicycles and a sign reading "Stretch Lids".

  **2. `Testing_Normal_Videos_Anomaly/Normal_Videos_891`**  *(anomaly_type: normal)*
   > The video shows a woman in a black dress seated at a table and engaged in conversation with a man in a blue shirt. A man in a white shirt walked past the table. Later, a man in a green shirt walked past the table. Throughout the video, the scene remained normal and uneventful.

  **3. `Testing_Normal_Videos_Anomaly/Normal_Videos_782`**  *(anomaly_type: potential threat)*
   > A man in a blue shirt and black pants approached a woman in a black dress and a man in a white shirt and black pants. Subsequently, a man in red ran towards the woman in black. A man in black clothes walked to the right and then to the left of the screen. These actions occurred within a room.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_758`**  *(anomaly_type: normal)*
   > The video shows a normal and uneventful scene at an indoor trade show or exhibition. A large crowd of people are walking around, browsing, and interacting with each other. Individuals are engaged in various activities, including conversing and examining merchandise. The scene is bustling with activity throughout the duration of the video.

  **5. `Testing_Normal_Videos_Anomaly/Normal_Videos_938`**  *(anomaly_type: none)*
   > The video shows a man in black and a woman in white entering a store. Subsequently, a man in white clothes leaves the store. Later, a woman in white shorts and a man in a white shirt enter the store, and a woman in a white shirt and blue shorts exits the store. Throughout the video, no unusual or suspicious events are observed.

### Stage 2 — Reranker v6 ck-50 top-5 (GT rank: 1)

★ **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_884`**  *(anomaly_type: none; rr_score=0.0903, stage1_score=0.3765)*
   > The video depicts a woman in a blue shirt entering a store and walking to the counter. She then exits the store. The scene shows a mundane activity with no unusual events. Multiple clips corroborate this description, showing a similar sequence of events.

  **2. `Shoplifting/Shoplifting028`**  *(anomaly_type: suspicious behavior; rr_score=0.0125, stage1_score=0.3327)*
   > A man wearing white clothes and a woman wearing green clothes repeatedly entered the store. In some clips, they were observed with a child. Other clips show them taking items from the shelves. In one clip, a man was seen holding a gun, pointing it at a woman.

  **3. `Testing_Normal_Videos_Anomaly/Normal_Videos_015`**  *(anomaly_type: none; rr_score=0.0103, stage1_score=0.3338)*
   > The video shows a woman in black clothes walking to the right side of the screen, and a man in black clothes walking to the left side of the screen. Both individuals move from the bottom to the top of the screen. There are no unusual or suspicious activities depicted in the video.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_224`**  *(anomaly_type: potential threat; rr_score=0.0100, stage1_score=0.3361)*
   > A man in a black shirt and helmet entered a room, sat down, and then left. The video depicts a mundane scene with no unusual events. Multiple clips describe a man in black entering a store or restaurant and then departing. One clip mentions a man holding a gun, but this detail is not corroborated by the other clips or the global caption.

  **5. `Shoplifting/Shoplifting007`**  *(anomaly_type: interaction; rr_score=0.0089, stage1_score=0.3368)*
   > A woman in a yellow dress and purple skirt entered a room, followed by a man in black clothes. The man approached the woman, and they exchanged a red garment. Subsequently, the man in black left the room. A woman in a green top and black pants grabbed a blue garment from the floor and placed it on a shelf.


---

## F1 — FAILURE — all stages miss top-30 (generic caption)

**Query:** *Two men came to the shop to exchange coupons for things.*

**Ground truth:** `Testing_Normal_Videos_Anomaly/Normal_Videos_929`

**GT video summary** (`anomaly_type`: *none*):
> The video shows a man in a black shirt walking to a counter. A woman in a black top and black headscarf is seated at the counter. The man then leaves the store. There are no unusual or suspicious activities observed.

### Stage 0 — ZS embedder top-5 (GT rank: miss)

  **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_417`**  *(anomaly_type: normal)*
   > The video depicts a man in a gray sweatshirt and jeans approaching a counter and interacting with an individual behind it. He then leaves the scene. Throughout the video, the man in the gray sweatshirt stands in front of the counter, observing a man working at a computer. The man in the gray sweatshirt appears to be waiting while the other man types on the computer.

  **2. `Testing_Normal_Videos_Anomaly/Normal_Videos_100`**  *(anomaly_type: normal)*
   > The video depicts a normal and mundane scene in a store. Several people, including men and women, are observed walking to the counter. Individuals are wearing various colored shirts and pants, such as blue, black, pink, and green. The scene shows people entering and moving within the store environment.

  **3. `Shoplifting/Shoplifting029`**  *(anomaly_type: normal)*
   > The video shows a man in a black coat entering a store. He looks around the interior. Subsequently, the man exits the store. The scene contains no unusual or out-of-the-ordinary events.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_312`**  *(anomaly_type: none)*
   > The video shows a man in a green jacket walking through a supermarket aisle, browsing products. A woman in a blue top and black pants walks behind him, and a man in a blue shirt walks in front of him. Both individuals are engaged in their own activities within the supermarket. The video concludes with the man in a green jacket leaving the scene.

  **5. `Shoplifting/Shoplifting033`**  *(anomaly_type: normal)*
   > The video shows a man in a blue shirt and khaki shorts entering a store and approaching a counter. He engages in conversation with a woman wearing a black top and a white hat, who is standing at the counter. The man then walks away from the counter, and the woman continues to talk to him. Subsequently, the man exits the store, and the woman remains at the counter.

### Stage 1 — FT embedder (P2 ck-900) top-5 (GT rank: miss)

  **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_417`**  *(anomaly_type: normal)*
   > The video depicts a man in a gray sweatshirt and jeans approaching a counter and interacting with an individual behind it. He then leaves the scene. Throughout the video, the man in the gray sweatshirt stands in front of the counter, observing a man working at a computer. The man in the gray sweatshirt appears to be waiting while the other man types on the computer.

  **2. `Testing_Normal_Videos_Anomaly/Normal_Videos_100`**  *(anomaly_type: normal)*
   > The video depicts a normal and mundane scene in a store. Several people, including men and women, are observed walking to the counter. Individuals are wearing various colored shirts and pants, such as blue, black, pink, and green. The scene shows people entering and moving within the store environment.

  **3. `Testing_Normal_Videos_Anomaly/Normal_Videos_247`**  *(anomaly_type: normal)*
   > The video shows a man in blue clothes and a woman in blue clothes walking to a counter. The man in blue holds a white paper and hands it to the woman in black clothes, who then places it on the counter. The woman in blue continues working at the counter, picking up a box and placing it on the counter. Throughout the video, no unusual events are observed.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_939`**  *(anomaly_type: none)*
   > The video shows a man in a pink shirt standing behind a counter. He is observed handling a small object, possibly a piece of paper or card, and opening a box. Subsequently, he takes a small, round object from the box and places it on the counter before looking at it. Throughout the video, no unusual or suspicious events are present.

  **5. `Testing_Normal_Videos_Anomaly/Normal_Videos_893`**  *(anomaly_type: none)*
   > The video shows a man in a blue jacket and a man in a white shirt entering a store. Subsequently, two men in blue shirts enter the store and look around. The individuals appear to be browsing the store. No unusual events are observed throughout the video.

### Stage 2 — Reranker v6 ck-50 top-5 (GT rank: miss)

  **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_100`**  *(anomaly_type: normal; rr_score=0.1709, stage1_score=0.4731)*
   > The video depicts a normal and mundane scene in a store. Several people, including men and women, are observed walking to the counter. Individuals are wearing various colored shirts and pants, such as blue, black, pink, and green. The scene shows people entering and moving within the store environment.

  **2. `Shoplifting/Shoplifting029`**  *(anomaly_type: normal; rr_score=0.1309, stage1_score=0.4066)*
   > The video shows a man in a black coat entering a store. He looks around the interior. Subsequently, the man exits the store. The scene contains no unusual or out-of-the-ordinary events.

  **3. `Shoplifting/Shoplifting005`**  *(anomaly_type: physical altercation; rr_score=0.1289, stage1_score=0.4187)*
   > The video begins with a man in a brown jacket entering and exiting a store. Subsequently, a man in a green jacket and black pants engaged in a physical altercation with another man. A man in a brown jacket approached a woman in a black coat and black pants, and then pushed her to the ground.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_312`**  *(anomaly_type: none; rr_score=0.1240, stage1_score=0.4137)*
   > The video shows a man in a green jacket walking through a supermarket aisle, browsing products. A woman in a blue top and black pants walks behind him, and a man in a blue shirt walks in front of him. Both individuals are engaged in their own activities within the supermarket. The video concludes with the man in a green jacket leaving the scene.

  **5. `Testing_Normal_Videos_Anomaly/Normal_Videos_059`**  *(anomaly_type: none; rr_score=0.1177, stage1_score=0.4263)*
   > The video shows a man in a blue shirt entering a store. He walked to a counter and then left the store. No unusual events were observed during the sequence. The video depicts a mundane scene with no suspicious activity.


---

## F2 — FAILURE — reranker degrades (ZS@10 → FT@1 → RR@20)

**Query:** *On the street, two strong men walked into the store, and an old man with white hair was walking on the street.*

**Ground truth:** `Testing_Normal_Videos_Anomaly/Normal_Videos_251`

**GT video summary** (`anomaly_type`: *normal*):
> The video shows a normal traffic scene with a red car parked on the side of the road and a white van driving past. A man in a white shirt and cap walks out of a store, and another man in a white shirt and black pants exits the store. A man in white clothes and a cap walks from the right side of the screen to the left side of the screen, and then back to the right side of the screen. Throughout the video, no unusual or out-of-the-ordinary events occur.

### Stage 0 — ZS embedder top-5 (GT rank: 10)

  **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_932`**  *(anomaly_type: None)*
   > A man wearing a red shirt and black pants entered the store. He walked to the counter and then left the store. Other men in black clothing were also observed entering and exiting the store. The scene appeared unremarkable.

  **2. `Shooting/Shooting028`**  *(anomaly_type: violent behavior)*
   > A man in black entered a room and approached a counter. He then threw a bottle at the counter. Subsequently, a man in black and white clothes ran out of the room. A man in black left the room.

  **3. `Testing_Normal_Videos_Anomaly/Normal_Videos_641`**  *(anomaly_type: normal)*
   > The video shows a man in blue and a woman in white in a supermarket. They engage in a conversation, with the man gesturing and the woman looking at him. The man then exits the store, and the woman continues to walk around the store. The scene consists of ordinary movements and actions within the supermarket environment.

  **4. `Shoplifting/Shoplifting031`**  *(anomaly_type: None)*
   > A man in a white shirt and black pants entered a store and approached the counter. He then left the store. Two men, one in a blue shirt and one in a white shirt, were sitting on the floor. Another man in a white shirt was standing at the counter. The scene appeared to be uneventful.

  **5. `Robbery/Robbery048`**  *(anomaly_type: potential physical altercation)*
   > A man in a striped shirt entered a store and then left. Another man in a white shirt entered the store. A man in a striped shirt and dark pants approached another man in a white shirt and black pants. The two men then moved away from each other.

### Stage 1 — FT embedder (P2 ck-900) top-5 (GT rank: 1)

★ **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_251`**  *(anomaly_type: normal)*
   > The video shows a normal traffic scene with a red car parked on the side of the road and a white van driving past. A man in a white shirt and cap walks out of a store, and another man in a white shirt and black pants exits the store. A man in white clothes and a cap walks from the right side of the screen to the left side of the screen, and then back to the right side of the screen. Throughout the video, no unusual or out-of-the-ordinary events occur.

  **2. `Testing_Normal_Videos_Anomaly/Normal_Videos_932`**  *(anomaly_type: None)*
   > A man wearing a red shirt and black pants entered the store. He walked to the counter and then left the store. Other men in black clothing were also observed entering and exiting the store. The scene appeared unremarkable.

  **3. `Testing_Normal_Videos_Anomaly/Normal_Videos_867`**  *(anomaly_type: unusual behavior)*
   > A man in white clothes walked towards a green truck and then left the scene. Another man in black clothes also walked out of the frame. The green truck and a motorcycle were parked on the road. There were no unusual events observed.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_686`**  *(anomaly_type: none)*
   > The video shows a man in a blue shirt and gray pants walking from one side of the road to the other. Other people are walking in the opposite direction. A building with a green roof and a large pile of sandbags is visible. A car drives by on the road.

  **5. `Testing_Normal_Videos_Anomaly/Normal_Videos_129`**  *(anomaly_type: unspecified)*
   > Two men walked towards the road, and then towards the police officers. One man wore orange clothing, another wore black clothing, and a third wore red clothing. The men moved in proximity to the police officers.

### Stage 2 — Reranker v6 ck-50 top-5 (GT rank: 20)

  **1. `Testing_Normal_Videos_Anomaly/Normal_Videos_867`**  *(anomaly_type: unusual behavior; rr_score=0.1562, stage1_score=0.4959)*
   > A man in white clothes walked towards a green truck and then left the scene. Another man in black clothes also walked out of the frame. The green truck and a motorcycle were parked on the road. There were no unusual events observed.

  **2. `Testing_Normal_Videos_Anomaly/Normal_Videos_686`**  *(anomaly_type: none; rr_score=0.1279, stage1_score=0.4803)*
   > The video shows a man in a blue shirt and gray pants walking from one side of the road to the other. Other people are walking in the opposite direction. A building with a green roof and a large pile of sandbags is visible. A car drives by on the road.

  **3. `Testing_Normal_Videos_Anomaly/Normal_Videos_129`**  *(anomaly_type: unspecified; rr_score=0.0454, stage1_score=0.4725)*
   > Two men walked towards the road, and then towards the police officers. One man wore orange clothing, another wore black clothing, and a third wore red clothing. The men moved in proximity to the police officers.

  **4. `Shooting/Shooting024`**  *(anomaly_type: violent encounter; rr_score=0.0396, stage1_score=0.4529)*
   > A man in a white shirt and blue jeans approached a woman in a blue dress and black skirt, then pushed her to the ground. Another man in white clothes entered the store. Subsequently, a man in blue clothes ran from the bottom to the top of the screen, and a woman in blue clothes did the same. These actions occurred near the store and in the street.

  **5. `Testing_Normal_Videos_Anomaly/Normal_Videos_896`**  *(anomaly_type: normal; rr_score=0.0374, stage1_score=0.4240)*
   > The video shows a normal traffic scene. A silver car drives past a red car. A woman in black clothes and a child in a red jacket are walking on the road. A man in black clothes is walking on the sidewalk, and another man in black clothes is pushing a stroller.

GT in RR ranking → rank **20** (rr_score=0.0097, stage1_score=0.5035)


---

## F3 — FAILURE — reranker keyword bias (ZS@1 → FT@1 → RR@10)

**Query:** *In the shop, a man in a hat who was about to rob was shot down by a man next to him.*

**Ground truth:** `Shooting/Shooting018`

**GT video summary** (`anomaly_type`: *physical altercation*):
> A man in a white shirt and black pants is holding a gun, and another man in green clothes is lying on the ground. A man in a green shirt and black pants pushed a man in a white shirt and blue jeans to the ground. A man in a green shirt and black pants approached and grabbed a man in a white shirt and black pants, causing him to fall to the ground. The man in a white shirt and black pants grabbed a man in black pants by the neck, causing him to fall.

### Stage 0 — ZS embedder top-5 (GT rank: 1)

★ **1. `Shooting/Shooting018`**  *(anomaly_type: physical altercation)*
   > A man in a white shirt and black pants is holding a gun, and another man in green clothes is lying on the ground. A man in a green shirt and black pants pushed a man in a white shirt and blue jeans to the ground. A man in a green shirt and black pants approached and grabbed a man in a white shirt and black pants, causing him to fall to the ground. The man in a white shirt and black pants grabbed a man in black pants by the neck, causing him to fall.

  **2. `Robbery/Robbery048`**  *(anomaly_type: potential physical altercation)*
   > A man in a striped shirt entered a store and then left. Another man in a white shirt entered the store. A man in a striped shirt and dark pants approached another man in a white shirt and black pants. The two men then moved away from each other.

  **3. `Shooting/Shooting028`**  *(anomaly_type: violent behavior)*
   > A man in black entered a room and approached a counter. He then threw a bottle at the counter. Subsequently, a man in black and white clothes ran out of the room. A man in black left the room.

  **4. `Testing_Normal_Videos_Anomaly/Normal_Videos_934`**  *(anomaly_type: potential robbery or theft)*
   > The video depicts a man wearing blue clothes walking from the bottom to the top of the screen and then returning to the bottom. Multiple clips show a man in black clothes walking from the bottom to the top of the screen and then returning to the bottom. One clip shows a man in blue clothes and a blue cap entering and exiting a store.

  **5. `Shoplifting/Shoplifting029`**  *(anomaly_type: normal)*
   > The video shows a man in a black coat entering a store. He looks around the interior. Subsequently, the man exits the store. The scene contains no unusual or out-of-the-ordinary events.

### Stage 1 — FT embedder (P2 ck-900) top-5 (GT rank: 1)

★ **1. `Shooting/Shooting018`**  *(anomaly_type: physical altercation)*
   > A man in a white shirt and black pants is holding a gun, and another man in green clothes is lying on the ground. A man in a green shirt and black pants pushed a man in a white shirt and blue jeans to the ground. A man in a green shirt and black pants approached and grabbed a man in a white shirt and black pants, causing him to fall to the ground. The man in a white shirt and black pants grabbed a man in black pants by the neck, causing him to fall.

  **2. `Robbery/Robbery048`**  *(anomaly_type: potential physical altercation)*
   > A man in a striped shirt entered a store and then left. Another man in a white shirt entered the store. A man in a striped shirt and dark pants approached another man in a white shirt and black pants. The two men then moved away from each other.

  **3. `Shooting/Shooting028`**  *(anomaly_type: violent behavior)*
   > A man in black entered a room and approached a counter. He then threw a bottle at the counter. Subsequently, a man in black and white clothes ran out of the room. A man in black left the room.

  **4. `Fighting/Fighting042`**  *(anomaly_type: violent encounter)*
   > A man in a black suit and hat entered a room and pointed a gun at another man in a black suit and hat. The second man then threw a gun from his hand. A man in black entered the room and pointed a gun at the man in black, who subsequently exited the room. The scene involved movement within a room.

  **5. `Shooting/Shooting011`**  *(anomaly_type: violent incident)*
   > A man in white clothes and a yellow helmet entered a room with white tiled floors and white chairs. He was holding a gun and pointing it at a man in black clothes, who then lay on the floor. Other individuals, including a man in blue clothes and a man in black clothes, were observed entering and exiting the room. Some individuals were restrained by others, with one individual holding a gun.

### Stage 2 — Reranker v6 ck-50 top-5 (GT rank: 10)

  **1. `Robbery/Robbery048`**  *(anomaly_type: potential physical altercation; rr_score=0.0718, stage1_score=0.5571)*
   > A man in a striped shirt entered a store and then left. Another man in a white shirt entered the store. A man in a striped shirt and dark pants approached another man in a white shirt and black pants. The two men then moved away from each other.

  **2. `Shoplifting/Shoplifting017`**  *(anomaly_type: none; rr_score=0.0645, stage1_score=0.4641)*
   > The video shows a man in blue clothes entering a store and walking to the counter. He then leaves the store. A woman in pink clothes also enters and leaves the store. The scene depicts a mundane event with no unusual occurrences.

  **3. `Shoplifting/Shoplifting049`**  *(anomaly_type: unusual item removal; rr_score=0.0601, stage1_score=0.3789)*
   > The video shows a man in a blue shirt entering a store. He takes an item from the counter and then leaves the store. Throughout the video, no unusual events are observed. The man is wearing light clothing.

  **4. `Shooting/Shooting008`**  *(anomaly_type: physical altercation; rr_score=0.0535, stage1_score=0.3950)*
   > A man in black clothes and a cap walked towards the camera, followed by a man in blue clothes and a cap. The man in black clothes picked up a black suitcase and walked away from the camera. A man in blue clothes and a cap moved a black suitcase from the lower right to the lower left corner of the video. A person in a white shirt and black pants exited a store and then left the store.

  **5. `Explosion/Explosion039`**  *(anomaly_type: unusual interaction; rr_score=0.0488, stage1_score=0.4228)*
   > A man wearing a blue shirt and pink hat entered a kitchen area and interacted with a cashier. He then left the kitchen area.  A man in a black shirt and blue jeans approached a table and grabbed a woman's hand. The interactions occurred within a restaurant setting.

GT in RR ranking → rank **10** (rr_score=0.0194, stage1_score=0.5676)

