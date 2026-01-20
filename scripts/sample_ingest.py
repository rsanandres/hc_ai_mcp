#!/usr/bin/env python3
"""Sample script demonstrating how to ingest FHIR data into the vector store.

This script shows how to:
1. Load FHIR bundle JSON files
2. Extract resources
3. Ingest them using the embeddings module

Usage:
    python scripts/sample_ingest.py /path/to/fhir_bundle.json
    python scripts/sample_ingest.py /path/to/fhir_bundles_directory/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


async def ingest_fhir_bundle(bundle_path: Path, patient_id: str | None = None) -> dict:
    """Ingest a FHIR bundle JSON file.
    
    Args:
        bundle_path: Path to the FHIR bundle JSON file.
        patient_id: Optional patient ID (extracted from bundle if not provided).
    
    Returns:
        Dictionary with ingest results.
    """
    from embeddings import process_and_store, IngestRequest
    
    print(f"\nProcessing: {bundle_path}")
    
    with open(bundle_path, "r") as f:
        bundle = json.load(f)
    
    # Extract patient ID from bundle if not provided
    if not patient_id:
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                patient_id = resource.get("id", "unknown")
                break
    
    if not patient_id:
        patient_id = "unknown"
    
    results = {
        "file": str(bundle_path),
        "patient_id": patient_id,
        "resources_processed": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
        "errors": [],
    }
    
    # Process each entry in the bundle
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "Unknown")
        resource_id = resource.get("id", "unknown")
        full_url = entry.get("fullUrl", "")
        
        # Skip certain resource types that don't need embedding
        skip_types = {"Bundle", "CapabilityStatement", "OperationOutcome"}
        if resource_type in skip_types:
            continue
        
        # Convert resource to text content
        content = _resource_to_text(resource)
        if not content or len(content.strip()) < 10:
            continue
        
        # Create ingest request
        request = IngestRequest(
            id=resource_id,
            resourceType=resource_type,
            content=content,
            patientId=patient_id,
            fullUrl=full_url,
            resourceJson=json.dumps(resource),
            sourceFile=str(bundle_path),
        )
        
        try:
            result = await process_and_store(request)
            results["resources_processed"] += 1
            results["chunks_created"] += result.chunks_created
            results["chunks_stored"] += result.chunks_stored
            if result.errors:
                results["errors"].extend(result.errors)
            
            print(f"  {resource_type}/{resource_id}: {result.chunks_stored}/{result.chunks_created} chunks")
        
        except Exception as e:
            results["errors"].append(f"{resource_type}/{resource_id}: {str(e)}")
            print(f"  {resource_type}/{resource_id}: ERROR - {e}")
    
    return results


def _resource_to_text(resource: dict) -> str:
    """Convert a FHIR resource to text for embedding.
    
    This is a simple implementation - customize based on your needs.
    """
    parts = []
    resource_type = resource.get("resourceType", "Unknown")
    
    # Add resource type
    parts.append(f"Resource Type: {resource_type}")
    
    # Add common fields
    if "status" in resource:
        parts.append(f"Status: {resource['status']}")
    
    # Handle different resource types
    if resource_type == "Observation":
        if "code" in resource and "text" in resource["code"]:
            parts.append(f"Observation: {resource['code']['text']}")
        if "valueQuantity" in resource:
            vq = resource["valueQuantity"]
            parts.append(f"Value: {vq.get('value', '')} {vq.get('unit', '')}")
        if "valueString" in resource:
            parts.append(f"Value: {resource['valueString']}")
    
    elif resource_type == "Condition":
        if "code" in resource and "text" in resource["code"]:
            parts.append(f"Condition: {resource['code']['text']}")
        if "clinicalStatus" in resource:
            parts.append(f"Clinical Status: {resource['clinicalStatus']}")
    
    elif resource_type == "MedicationRequest":
        if "medicationCodeableConcept" in resource:
            med = resource["medicationCodeableConcept"]
            parts.append(f"Medication: {med.get('text', '')}")
        if "dosageInstruction" in resource:
            for dosage in resource["dosageInstruction"]:
                if "text" in dosage:
                    parts.append(f"Dosage: {dosage['text']}")
    
    elif resource_type == "DiagnosticReport":
        if "code" in resource and "text" in resource["code"]:
            parts.append(f"Report: {resource['code']['text']}")
        if "conclusion" in resource:
            parts.append(f"Conclusion: {resource['conclusion']}")
    
    elif resource_type == "Patient":
        if "name" in resource and resource["name"]:
            name = resource["name"][0]
            parts.append(f"Patient: {name.get('given', [''])[0]} {name.get('family', '')}")
        if "birthDate" in resource:
            parts.append(f"Birth Date: {resource['birthDate']}")
        if "gender" in resource:
            parts.append(f"Gender: {resource['gender']}")
    
    # Add any text field
    if "text" in resource and "div" in resource["text"]:
        # Strip HTML tags (simple approach)
        import re
        text = re.sub(r"<[^>]+>", " ", resource["text"]["div"])
        text = " ".join(text.split())
        if text:
            parts.append(f"Text: {text}")
    
    return "\n".join(parts)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest FHIR bundles into the vector store",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to FHIR bundle JSON file or directory containing bundles",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Override patient ID for all resources",
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    # Collect files to process
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("*.json"))
        if not files:
            print(f"No JSON files found in: {path}")
            sys.exit(1)
    
    print(f"Found {len(files)} file(s) to process")
    
    # Process each file
    total_results = {
        "files_processed": 0,
        "resources_processed": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
        "errors": [],
    }
    
    for file_path in files:
        try:
            result = await ingest_fhir_bundle(file_path, args.patient_id)
            total_results["files_processed"] += 1
            total_results["resources_processed"] += result["resources_processed"]
            total_results["chunks_created"] += result["chunks_created"]
            total_results["chunks_stored"] += result["chunks_stored"]
            total_results["errors"].extend(result["errors"])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            total_results["errors"].append(f"{file_path}: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Files processed:     {total_results['files_processed']}")
    print(f"Resources processed: {total_results['resources_processed']}")
    print(f"Chunks created:      {total_results['chunks_created']}")
    print(f"Chunks stored:       {total_results['chunks_stored']}")
    print(f"Errors:              {len(total_results['errors'])}")
    
    if total_results["errors"]:
        print("\nErrors:")
        for error in total_results["errors"][:10]:
            print(f"  - {error}")
        if len(total_results["errors"]) > 10:
            print(f"  ... and {len(total_results['errors']) - 10} more")


if __name__ == "__main__":
    asyncio.run(main())
