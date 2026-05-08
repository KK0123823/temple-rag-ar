INSERT INTO temples (temple_id, name_zh, city, address, founded_year, description)
VALUES ('00000000-0000-0000-0000-000000000001','示例宮','台南市','示例路 1 號',1850,'本宮主祀天上聖母。');

INSERT INTO deities (deity_id, name_zh, aliases, domain_of_influence, origin_myth) VALUES
('00000000-0000-0000-0000-000000000101','天上聖母','["媽祖","天妃"]','海上守護','傳說守護海民');

INSERT INTO temple_deity VALUES
('00000000-0000-0000-0000-000000000001','00000000-0000-0000-0000-000000000101','主祀');

INSERT INTO events (event_id, name_zh, event_type, lunar_rule, duration_days)
VALUES ('00000000-0000-0000-0000-000000000201','示例宮年度繞境','繞境','農曆三月二十三',2);

INSERT INTO sources (source_id, title, year, type, license)
VALUES ('00000000-0000-0000-0000-000000000301','示例宮沿革節錄',2020,'text','CC BY-NC');

INSERT INTO documents (doc_id, source_id, media_type, language, text_content)
VALUES ('00000000-0000-0000-0000-000000000401','00000000-0000-0000-0000-000000000301','text','zh-TW',
'本宮主祀天上聖母，年度繞境於農曆三月舉行，起駕於前殿，沿途設香案迎駕。');
