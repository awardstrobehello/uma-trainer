-- Table: public.support_cards

-- DROP TABLE IF EXISTS public.support_cards;

CREATE TABLE IF NOT EXISTS public.support_cards
(
    id integer NOT NULL DEFAULT nextval('support_cards_id_seq'::regclass),
    name character varying(100) COLLATE pg_catalog."default" NOT NULL,
    type character varying(20) COLLATE pg_catalog."default" NOT NULL,
    rarity character varying(3) COLLATE pg_catalog."default" NOT NULL,
    limit_break integer NOT NULL,
    friendship_bonus integer DEFAULT 0,
    training_effectiveness integer DEFAULT 0,
    mood_effect integer DEFAULT 0,
    initial_friendship integer DEFAULT 0,
    speed_bonus integer DEFAULT 0,
    stamina_bonus integer DEFAULT 0,
    power_bonus integer DEFAULT 0,
    guts_bonus integer DEFAULT 0,
    wit_bonus integer DEFAULT 0,
    specialty_priority integer DEFAULT 0,
    skill_bonus integer DEFAULT 0,
    race_bonus integer DEFAULT 0,
    fan_bonus integer DEFAULT 0,
    hint_frequency integer DEFAULT 0,
    hint_level integer DEFAULT 0,
    CONSTRAINT support_cards_pkey PRIMARY KEY (id),
    CONSTRAINT support_cards_type_check CHECK (type::text = ANY (ARRAY['Speed'::character varying, 'Stamina'::character varying, 'Power'::character varying, 'Guts'::character varying, 'Wit'::character varying, 'Friend'::character varying, 'Group'::character varying]::text[])),
    CONSTRAINT support_cards_rarity_check CHECK (rarity::text = ANY (ARRAY['R'::character varying, 'SR'::character varying, 'SSR'::character varying]::text[])),
    CONSTRAINT support_cards_limit_break_check CHECK (limit_break >= 0 AND limit_break <= 4)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.support_cards
    OWNER to postgres;