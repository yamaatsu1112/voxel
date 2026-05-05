use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use std::collections::HashSet;
use syn::Type;

/// Implementation logic for impl_query_item_tuple macro
pub fn impl_query_item_tuple_impl(input: TokenStream) -> TokenStream {
    // Convert proc_macro::TokenStream to proc_macro2::TokenStream for internal processing
    let input2: proc_macro2::TokenStream = input.into();

    // Parse using syn::parse2 (works with proc_macro2::TokenStream)
    let ty = match syn::parse2::<Type>(input2) {
        Ok(ty) => ty,
        Err(e) => return e.to_compile_error().into(),
    };

    let Type::Tuple(tuple) = &ty else {
        return syn::Error::new_spanned(&ty, "Expected a tuple type")
            .to_compile_error()
            .into();
    };

    let elems = &tuple.elems;
    let len = elems.len();

    if len == 0 {
        return syn::Error::new_spanned(tuple, "Tuple must have at least one element")
            .to_compile_error()
            .into();
    }

    // Extract component types and mutability information
    let mut component_types = Vec::new();
    let mut item_types = Vec::new();
    let mut item_mut_types = Vec::new();
    let mut fetch_calls = Vec::new();
    let mut fetch_mut_calls = Vec::new();

    for (i, elem) in elems.iter().enumerate() {
        let (component_type, is_mut) = extract_component_type(elem);
        component_types.push(component_type.clone());

        // For Item<'w>, always use &'w T (immutable reference)
        item_types.push(quote! { &'w #component_type });

        // For ItemMut<'w>, use &'w mut T if original was &mut, otherwise &'w T
        if is_mut {
            item_mut_types.push(quote! { &'w mut #component_type });
        } else {
            item_mut_types.push(quote! { &'w #component_type });
        }

        // Generate fetch calls
        let var_name = syn::Ident::new(&format!("item_{}", i), Span::call_site());
        fetch_calls.push(quote! {
            let #var_name = archetype.get_component_at::<#component_type>(index)?;
        });

        // Generate fetch_mut calls
        fetch_mut_calls.push(quote! {
            let #var_name = (*archetype_ptr).get_component_mut_at::<#component_type>(index)?;
        });
    }

    // Build tuple types
    let item_tuple = quote! { (#(#item_types),*) };
    let item_mut_tuple = quote! { (#(#item_mut_types),*) };

    // Build fetch return tuple
    let fetch_tuple_items: Vec<_> = (0..len)
        .map(|i| syn::Ident::new(&format!("item_{}", i), Span::call_site()))
        .collect();
    let fetch_return = quote! { Some((#(#fetch_tuple_items),*)) };

    // Build fetch_mut return tuple
    let fetch_mut_return = quote! { Some((#(#fetch_tuple_items),*)) };

    // Generate type IDs for archetypes
    let type_id_calls: Vec<_> = component_types
        .iter()
        .map(|ty| quote! { std::any::TypeId::of::<#ty>() })
        .collect();

    // Extract type parameters from component types
    // For types like &T1, we need to extract T1 as a type parameter
    let mut type_params = Vec::new();
    let mut where_clauses = Vec::new();
    let mut seen_params = HashSet::new();

    for component_type in &component_types {
        // Check if the type is a path type (like T1, T2, etc.)
        if let Type::Path(type_path) = component_type
            && let Some(segment) = type_path.path.segments.last()
        {
            // Check if this is a simple identifier (type parameter)
            if type_path.path.segments.len() == 1 && segment.arguments.is_empty() {
                let param_name = &segment.ident;
                let param_str = param_name.to_string();

                // Avoid duplicates
                if seen_params.insert(param_str) {
                    type_params.push(quote! { #param_name });
                    where_clauses.push(quote! { #param_name: Component + 'static });
                }
            }
        }
    }

    // Generate impl block with type parameters if we found any
    let impl_block = if !type_params.is_empty() {
        // Generate type parameter list: <T1, T2>
        let type_param_list = quote! { <#(#type_params),*> };
        quote! {
            impl #type_param_list QueryItem for #ty
            where
                #(#where_clauses),*
        }
    } else {
        // No type parameters found, generate where clauses for concrete types
        let concrete_where_clauses: Vec<_> = component_types
            .iter()
            .map(|ty| quote! { #ty: Component + 'static })
            .collect();
        quote! {
            impl QueryItem for #ty
            where
                #(#concrete_where_clauses),*
        }
    };

    let expanded = quote! {
        #impl_block
        {
            type Item<'w> = #item_tuple;
            type ItemMut<'w> = #item_mut_tuple;

            fn fetch<'w>(archetype: &'w Archetype, index: usize) -> Option<Self::Item<'w>> {
                #(#fetch_calls)*
                #fetch_return
            }

            fn fetch_mut<'w>(archetype: &'w mut Archetype, index: usize) -> Option<Self::ItemMut<'w>> {
                let archetype_ptr = archetype as *mut Archetype;
                unsafe {
                    #(#fetch_mut_calls)*
                    #fetch_mut_return
                }
            }

            fn archetypes<'w>(world: &'w World) -> Vec<ArchetypeId> {
                world.archetypes_by_type_ids(&vec![#(#type_id_calls),*])
            }
        }
    };

    expanded.into()
}

/// Extract component type and mutability from a reference type
fn extract_component_type(ty: &Type) -> (Type, bool) {
    match ty {
        Type::Reference(ref_ty) => {
            let is_mut = ref_ty.mutability.is_some();
            let inner_ty = ref_ty.elem.as_ref().clone();
            (inner_ty, is_mut)
        }
        _ => {
            // If it's not a reference, assume it's the component type itself
            (ty.clone(), false)
        }
    }
}
