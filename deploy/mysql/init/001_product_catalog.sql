SET NAMES utf8mb4;
SET time_zone = '+08:00';

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    applied_at DATETIME(6) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS manufacturers (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    website TEXT,
    country_code VARCHAR(16),
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS source_references (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    document_title VARCHAR(512) NOT NULL,
    source_path TEXT NOT NULL,
    source_url TEXT,
    source_sha256 CHAR(64) NOT NULL,
    page_number INT NOT NULL CHECK (page_number > 0),
    retrieved_at DATETIME(6),
    created_at DATETIME(6) NOT NULL,
    UNIQUE KEY uq_source_reference(document_title, source_sha256, page_number),
    KEY idx_source_references_hash(source_sha256, page_number)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS products (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    manufacturer_id BIGINT NOT NULL,
    model VARCHAR(255) NOT NULL,
    name VARCHAR(512) NOT NULL,
    modality ENUM('pure_infrared', 'pure_visible', 'dual_ir_visible') NOT NULL,
    summary TEXT NOT NULL,
    lifecycle_status ENUM('draft', 'active', 'retired') NOT NULL DEFAULT 'active',
    primary_source_ref_id BIGINT,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    UNIQUE KEY uq_product_model(manufacturer_id, model),
    KEY idx_products_modality_status(modality, lifecycle_status),
    CONSTRAINT fk_products_manufacturer FOREIGN KEY(manufacturer_id) REFERENCES manufacturers(id),
    CONSTRAINT fk_products_source FOREIGN KEY(primary_source_ref_id) REFERENCES source_references(id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS spec_definitions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    spec_key VARCHAR(128) NOT NULL UNIQUE,
    label VARCHAR(255) NOT NULL,
    value_type ENUM('text', 'number', 'boolean', 'json') NOT NULL,
    category VARCHAR(128) NOT NULL DEFAULT 'general',
    default_unit VARCHAR(64),
    description TEXT NOT NULL,
    created_at DATETIME(6) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS product_spec_values (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    spec_definition_id BIGINT NOT NULL,
    value_text TEXT,
    value_number DOUBLE,
    value_boolean BOOLEAN,
    value_json JSON,
    unit VARCHAR(64),
    comparator ENUM('eq', 'lt', 'lte', 'gt', 'gte', 'range'),
    source_ref_id BIGINT NOT NULL,
    source_excerpt TEXT NOT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    UNIQUE KEY uq_product_spec(product_id, spec_definition_id),
    KEY idx_product_spec_number(spec_definition_id, value_number),
    KEY idx_product_spec_boolean(spec_definition_id, value_boolean),
    CONSTRAINT fk_spec_value_product FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    CONSTRAINT fk_spec_value_definition FOREIGN KEY(spec_definition_id) REFERENCES spec_definitions(id),
    CONSTRAINT fk_spec_value_source FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS product_prices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    price_type ENUM('public_list', 'dealer_reference', 'historical', 'inquiry', 'unknown') NOT NULL,
    amount_min DOUBLE,
    amount_max DOUBLE,
    currency VARCHAR(16),
    tax_included BOOLEAN,
    effective_from DATE,
    effective_to DATE,
    source_ref_id BIGINT NOT NULL,
    note TEXT NOT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    KEY idx_product_prices_product(product_id, price_type),
    CONSTRAINT fk_price_product FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    CONSTRAINT fk_price_source FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS product_media (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    media_type ENUM('product_image', 'datasheet_page', 'sample_image', 'other') NOT NULL,
    uri TEXT NOT NULL,
    uri_hash BINARY(32) GENERATED ALWAYS AS (UNHEX(SHA2(uri, 256))) STORED,
    caption TEXT NOT NULL,
    source_ref_id BIGINT NOT NULL,
    sort_order INT NOT NULL DEFAULT 0,
    created_at DATETIME(6) NOT NULL,
    UNIQUE KEY uq_product_media(product_id, uri_hash),
    CONSTRAINT fk_media_product FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    CONSTRAINT fk_media_source FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS product_scenarios (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    scenario VARCHAR(255) NOT NULL,
    suitability ENUM('supported', 'conditional', 'not_recommended') NOT NULL DEFAULT 'supported',
    note TEXT NOT NULL,
    source_ref_id BIGINT NOT NULL,
    created_at DATETIME(6) NOT NULL,
    UNIQUE KEY uq_product_scenario(product_id, scenario),
    CONSTRAINT fk_scenario_product FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    CONSTRAINT fk_scenario_source FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS fusion_compatibility (
    product_id BIGINT PRIMARY KEY,
    mambadfuse_role ENUM('infrared_source', 'visible_source', 'dual_source_candidate', 'not_applicable') NOT NULL,
    infrared_stream_access BOOLEAN,
    visible_stream_access BOOLEAN,
    independent_modal_streams BOOLEAN,
    hardware_trigger BOOLEAN,
    timestamp_sync BOOLEAN,
    sdk_access BOOLEAN,
    calibration_data BOOLEAN,
    registration_required BOOLEAN NOT NULL,
    suitability_status ENUM('suitable', 'conditional', 'not_suitable', 'unknown') NOT NULL,
    notes TEXT NOT NULL,
    source_ref_id BIGINT NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_fusion_product FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    CONSTRAINT fk_fusion_source FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS product_spec_candidates (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    document_table_id BIGINT,
    source_document_title VARCHAR(512) NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 CHAR(64) NOT NULL,
    source_page INT NOT NULL CHECK (source_page > 0),
    source_table_index INT NOT NULL CHECK (source_table_index > 0),
    manufacturer_name VARCHAR(255) NOT NULL,
    model VARCHAR(255) NOT NULL,
    product_name VARCHAR(512) NOT NULL,
    modality ENUM('pure_infrared', 'pure_visible', 'dual_ir_visible') NOT NULL,
    spec_key VARCHAR(128) NOT NULL,
    spec_label VARCHAR(255) NOT NULL,
    category VARCHAR(128) NOT NULL DEFAULT 'general',
    value_type ENUM('text', 'number', 'boolean', 'json') NOT NULL,
    raw_value TEXT NOT NULL,
    normalized_value_json JSON NOT NULL,
    unit VARCHAR(64),
    comparator ENUM('eq', 'lt', 'lte', 'gt', 'gte', 'range'),
    raw_row_json JSON NOT NULL,
    confidence DOUBLE NOT NULL DEFAULT 0.8 CHECK (confidence >= 0 AND confidence <= 1),
    review_status ENUM('pending', 'approved', 'rejected', 'published') NOT NULL DEFAULT 'pending',
    review_note TEXT NOT NULL,
    reviewed_by VARCHAR(255),
    reviewed_at DATETIME(6),
    published_product_id BIGINT,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    UNIQUE KEY uq_product_candidate(source_sha256, source_page, source_table_index, model, spec_key),
    KEY idx_product_candidates_status(review_status, model),
    KEY idx_product_candidates_source(source_sha256, source_page, source_table_index),
    CONSTRAINT fk_candidate_product FOREIGN KEY(published_product_id) REFERENCES products(id) ON DELETE SET NULL
) ENGINE=InnoDB;

INSERT IGNORE INTO schema_migrations(version, name, applied_at) VALUES
    (1, 'structured_product_catalog', NOW(6)),
    (2, 'product_spec_review_workflow', NOW(6)),
    (3, 'mysql84_json_and_enterprise_indexes', NOW(6));

